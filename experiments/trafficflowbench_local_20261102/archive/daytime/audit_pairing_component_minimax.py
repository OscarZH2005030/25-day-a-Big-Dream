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
    out=d.OUT/'task4_pairing_component_minimax'
    baseline=getattr(d,'FROZEN_TASK4_BASELINE',None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        predictor_splits=['train','validation'],evaluation_split='private',
        method='Minimize the largest cell movement among near-optimal L1 solutions under the same constraints.'))
    frozen=pd.read_csv(baseline,dtype={'panel':str,'departure_time':str,'path_id':str})
    rows,hashes,failures=[],{},[]
    for panel in sorted(d.night.predictor.load_routes()):
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
        for view,matrix in [('margins',np.vstack((counts,groups))),
                            ('margins_and_histogram',np.vstack((counts,groups,bins)))]:
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
            cap = .05
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
                for selection, flow in [('primary', result.x[:n]), ('minimax', secondary.x[:n])]:
                    delta = flow - start
                    movement = np.abs(delta).sum() / total
                    prior_delta = (np.abs(flow-prior).sum() - prior_budget) / total
                    drift = np.max(np.abs(matrix @ delta))
                    error = np.abs(flow - ideal).sum() / total
                    assert drift < 1e-5 and flow.min() >= -1e-7 and movement <= cap + 1e-8 and prior_delta <= 1e-8
                    assert error <= result.fun / total + 2e-8
                    proxy_delta = (flow - proxy) / total
                    rows.append(dict(panel=panel, view=view, method=method, selection=selection,
                        movement=movement, max_cell_move=np.max(np.abs(delta)) / total,
                        normalized_proxy_gain=baseline_error-error, prior_distance_delta=prior_delta,
                        projected_prior_l1=np.abs(proxy_delta).sum(),
                        projected_prior_squared=proxy_delta @ proxy_delta, max_drift=drift))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['view','method','selection'])[['normalized_proxy_gain','movement','max_cell_move','prior_distance_delta','projected_prior_l1','projected_prior_squared']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
