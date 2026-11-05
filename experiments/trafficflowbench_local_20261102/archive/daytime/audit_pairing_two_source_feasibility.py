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
    out=d.OUT/'task4_pairing_two_source_feasibility'
    baseline=getattr(d,'FROZEN_TASK4_BASELINE',None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        predictor_splits=['train','validation'],evaluation_split='private',shrinkage=.5,
        method='Compare direct feasible interpolation with L1 projection onto baseline observation constraints. No draws or hidden labels.'))
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
        target=.5*np.mean(sources,axis=0)*total
        n=len(start)
        identity=sparse.eye(n,format='csr')
        inequalities=sparse.vstack((sparse.hstack((identity,-identity)),sparse.hstack((-identity,-identity))),format='csr')
        objective=np.r_[np.zeros(n),np.ones(n)]
        groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T for k in ('origin_zone','destination_zone')])
        origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                            for k in ('origin_zone','destination_zone')]
        span=destination-origin
        bins=(span[:,None]==np.unique(span)).astype(float).T
        for view,matrix in [('margins',np.vstack((counts,groups))),
                            ('margins_and_histogram',np.vstack((counts,groups,bins)))]:
            basis=orth(matrix.T,rcond=1e-10)
            delta=target-start
            direction=delta-basis@(basis.T@delta)
            ideal=start+direction
            negative=direction < -1e-10
            scale=min(1.,float(np.min(-start[negative]/direction[negative]))) if negative.any() else 1.
            interpolate=start+scale*direction
            equality=sparse.csr_matrix(matrix)
            equalities=sparse.hstack((equality,sparse.csr_matrix(equality.shape)),format='csr')
            result=linprog(objective,A_ub=inequalities,b_ub=np.r_[ideal,-ideal],
                A_eq=equalities,b_eq=equality@start,method='highs')
            if not result.success:
                failures.append(dict(panel=panel,view=view,error=result.message))
                continue
            for method,flow in [('baseline',start),('interpolate',interpolate),('l1_projection',result.x[:n])]:
                drift=np.max(np.abs(matrix@(flow-start)))
                assert drift<1e-5 and flow.min()>=-1e-7
                rows.append(dict(panel=panel,view=view,method=method,straight_step=scale,
                    ideal_negative_cells=int((ideal < -1e-9).sum()),distance_to_ideal=np.abs(flow-ideal).sum()/total,
                    movement=np.abs(flow-start).sum()/total,
                    prior_distance_delta=(np.abs(flow-prior).sum()-np.abs(start-prior).sum())/total,max_drift=drift))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['view','method'])[['distance_to_ideal','movement','prior_distance_delta']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
