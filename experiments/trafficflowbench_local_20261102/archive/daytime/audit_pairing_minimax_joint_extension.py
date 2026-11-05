"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth
from scipy import sparse
from scipy.optimize import linprog

import common as d
from audit_direct_od_measurement_design import base






def main():
    out=d.OUT/'task4_pairing_minimax_joint_extension'
    baseline=getattr(d,'FROZEN_TASK4_BASELINE',None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        predictor_splits=['train','validation'],evaluation_split='private',
        method='Find the smallest common minimax multiplier allowing equal squared-span moments and long-trip shares.'))
    frozen=pd.read_csv(baseline,dtype={'panel':str,'departure_time':str,'path_id':str})
    rows,hashes,failures=[],{},[]
    for panel in sorted(set(d.night.predictor.load_routes()) - {'D12_I405_N', 'D7_I10_E'}):
        paths,counts,_,prior,provenance=base.load(panel,'private')
        hashes.update(provenance)
        sources=[]
        for split in ('train','validation'):
            source_paths,_,_,source,provenance=base.load(panel,split)
            assert paths.path_id.equals(source_paths.path_id)
            hashes.update(provenance)
            sources.append(source/source.sum())
        departure=pd.read_csv(d.DATA/'task4'/panel/'private'/'synthetic_weak_prior.csv',dtype=str).departure_time.iloc[0]
        start=frozen.query('panel==@panel and departure_time==@departure').set_index('path_id').reindex(paths.path_id).path_flow.to_numpy()
        assert np.isfinite(start).all() and start.min()>=0
        total=start.sum()
        pooled=np.mean(sources,axis=0)*total
        n=len(start)
        identity=sparse.eye(n,format='csr')
        zero=sparse.csr_matrix(identity.shape)
        objective=np.r_[np.zeros(n),np.ones(n),np.zeros(2*n)]
        bounds_blocks=[]
        for block in range(3):
            pieces=[zero]*3
            pieces[block]=-identity
            bounds_blocks.extend((sparse.hstack((identity,*pieces)),sparse.hstack((-identity,*pieces))))
        prior_budget=np.abs(start-prior).sum()
        for block in (1,2):
            row=np.zeros(4*n)
            row[(block+1)*n:(block+2)*n]=1
            bounds_blocks.append(sparse.csr_matrix(row[None,:]))
        inequalities=sparse.vstack(bounds_blocks,format='csr')
        groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T for k in ('origin_zone','destination_zone')])
        origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                            for k in ('origin_zone','destination_zone')]
        span=destination-origin
        bins=(span[:,None]==np.unique(span)).astype(float).T
        for view,matrix in [('margins',np.vstack((counts,groups)))]:
            basis=orth(matrix.T,rcond=1e-10)
            observable = basis @ (basis.T @ start)
            paired = pooled - basis @ (basis.T @ pooled)
            hist_basis = orth(np.vstack((counts, groups, bins)).T,rcond=1e-10)
            free = pooled - hist_basis @ (hist_basis.T @ pooled)
            histogram = paired - free
            proxy = prior * (total / prior.sum())
            proxy = observable + proxy - basis @ (basis.T @ proxy)
            equality=sparse.csr_matrix(matrix)
            equalities=sparse.hstack((equality,sparse.csr_matrix((equality.shape[0],3*n))),format='csr')
            cap = .01
            systems = []
            for method, target in [('half', .5 * paired), ('fixed_separate', .5 * free + histogram)]:
                ideal = observable + target - basis @ (basis.T @ target)
                baseline_error = np.abs(start - ideal).sum() / total
                result=linprog(objective,A_ub=inequalities,b_ub=np.r_[ideal,-ideal,start,-start,prior,-prior,cap*total,prior_budget],
                    A_eq=equalities,b_eq=equality@start,method='highs')
                if not result.success:
                    failures.append(dict(panel=panel,view=view,method=method,cap=cap,error=result.message))
                    continue
                upper = np.r_[ideal, -ideal, start, -start, prior, -prior, cap * total, prior_budget]
                zero_n = sparse.csr_matrix((n, n))
                movement_block = sparse.hstack((zero_n, zero_n, identity, zero_n,
                                                 -np.ones((n, 1))), format='csr')
                secondary_ub = sparse.vstack((
                    sparse.hstack((inequalities, sparse.csr_matrix((inequalities.shape[0], 1)))),
                    sparse.csr_matrix(np.r_[objective, 0.][None, :]), movement_block), format='csr')
                secondary_eq = sparse.hstack((equalities, sparse.csr_matrix((equalities.shape[0], 1))),
                                             format='csr')
                secondary = linprog(np.r_[np.zeros(4*n), 1.], A_ub=secondary_ub,
                                    b_ub=np.r_[upper, result.fun + 1e-8 * total, np.zeros(n)],
                                    A_eq=secondary_eq, b_eq=equality @ start, method='highs')
                if not secondary.success:
                    failures.append(dict(panel=panel, view=view, method=method, error=secondary.message))
                    continue
                systems.append((secondary_ub, np.r_[upper, result.fun + 1e-8 * total, np.zeros(n)],
                                secondary_eq, equality @ start, result.fun, secondary.fun, ideal))
            if len(systems) != 2:
                continue
            size = 4*n+1
            span_cost = (span / np.max(np.abs(span))) ** 2
            ub = sparse.block_diag([item[0] for item in systems], format='csr')
            ub = sparse.hstack((ub, sparse.csr_matrix((ub.shape[0], 1))), format='csr')
            eq = sparse.block_diag([item[2] for item in systems], format='csr')
            eq = sparse.hstack((eq, sparse.csr_matrix((eq.shape[0], 1))), format='csr')
            span_row = np.r_[span_cost, np.zeros(size-n), -span_cost, np.zeros(size-n+1)]
            long_trip = (span > np.median(span)).astype(float)
            long_row = np.r_[long_trip, np.zeros(size-n), -long_trip, np.zeros(size-n+1)]
            eq = sparse.vstack((eq, sparse.csr_matrix(span_row[None, :]),
                                sparse.csr_matrix(long_row[None, :])), format='csr')
            link = np.zeros((2, 2*size+1))
            for index, item in enumerate(systems):
                link[index, (index+1)*size-1] = 1. / item[5]
                link[index, -1] = -1.
            ub = sparse.vstack((ub, sparse.csr_matrix(link)), format='csr')
            b_ub = np.r_[systems[0][1], systems[1][1], [1e-8 * total / item[5] for item in systems]]
            b_eq = np.r_[systems[0][3], systems[1][3], 0., 0.]
            overlap = linprog(np.r_[np.zeros(2*size), 1.], A_ub=ub, b_ub=b_ub,
                              A_eq=eq, b_eq=b_eq, bounds=[(0,None)]*(2*size)+[(1,None)], method='highs')
            if not overlap.success:
                failures.append(dict(panel=panel, view=view, error=overlap.message))
                continue
            for index, method in enumerate(('half', 'fixed_separate')):
                flow = overlap.x[index*size:index*size+n]
                delta = flow - start
                _, _, _, _, primary_opt, max_opt, ideal = systems[index]
                assert np.abs(delta).sum() <= (cap + 1e-8) * total
                assert np.abs(flow-prior).sum() <= prior_budget + 1e-8 * total
                assert np.abs(flow-ideal).sum() <= primary_opt + 2e-8 * total
                assert np.abs(delta).max() <= overlap.fun * max_opt + 2e-8 * total
                assert np.max(np.abs(matrix @ delta)) < 1e-5
                rows.append(dict(panel=panel, view=view, method=method, multiplier=overlap.fun,
                                 common_span_change=span_cost @ delta / total, common_long_trip_change=long_trip @ delta / total,
                                 max_cell_move=np.abs(delta).max() / total))
            for metric in ('common_span_change', 'common_long_trip_change'):
                assert abs(rows[-2][metric] - rows[-1][metric]) < 1e-9
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby('panel')[['multiplier','common_span_change','common_long_trip_change']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
