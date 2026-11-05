"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

import common as d
from audit_direct_od_measurement_design import base





def main():
    out=d.OUT/'task4_pairing_minimum_prior_ranges'
    baseline=getattr(d, 'FROZEN_TASK4_BASELINE', None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        splits=['validation','private'],quantiles=[.5],
        method='Compare median span bounds at minimum feasible L1 distance and at the midpoint to the frozen baseline L1 distance. Minimum gets a numerical slack of 1e-8 times fixed total demand. No draws.'))
    frozen=pd.read_csv(baseline,dtype={'panel':str,'departure_time':str,'path_id':str})
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        for split in ('validation','private'):
            paths, counts, _, prior, provenance=base.load(panel,split)
            hashes.update(provenance)
            file=d.DATA/'task4'/panel/split/'synthetic_weak_prior.csv'
            departure=pd.read_csv(file,dtype=str).departure_time.iloc[0]
            start=frozen.query('panel==@panel and departure_time==@departure').set_index('path_id').reindex(paths.path_id).path_flow.to_numpy()
            assert np.isfinite(start).all() and (start>=0).all()
            groups=np.vstack([pd.get_dummies(paths[key]).to_numpy(dtype=float).T for key in ('origin_zone','destination_zone')])
            oz,dz=[paths[key].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                   for key in ('origin_zone','destination_zone')]
            span=dz-oz
            order=np.argsort(span,kind='stable')
            threshold=span[order[np.searchsorted(np.cumsum(prior[order])/prior.sum(),.5)]]
            query=(span>threshold).astype(float)
            constraints=sparse.csr_matrix(np.vstack((counts,groups)))
            rhs=constraints@start
            n=len(paths)
            identity=sparse.eye(n,format='csr')
            l1_budget=np.abs(start-prior).sum()
            augmented_eq=sparse.hstack((constraints,sparse.csr_matrix(constraints.shape)),format='csr')
            budget_row=sparse.csr_matrix(np.r_[np.zeros(n),np.ones(n)][None,:])
            budget_matrix=sparse.vstack((sparse.hstack((identity,-identity)),sparse.hstack((-identity,-identity)),budget_row),format='csr')
            budget_rhs=np.r_[prior,-prior,l1_budget]
            minimum=linprog(np.r_[np.zeros(n),np.ones(n)],A_eq=augmented_eq,b_eq=rhs,
                A_ub=budget_matrix[:-1],b_ub=budget_rhs[:-1],method='highs')
            if not minimum.success:
                failures.append(dict(panel=panel,split=split,regime='minimum_l1',error=minimum.message))
                continue
            slack=1e-8*start.sum()
            assert minimum.fun<=l1_budget+slack
            for regime,budget in [('minimum',min(l1_budget,minimum.fun+slack)),('midpoint',.5*(l1_budget+minimum.fun))]:
                budget_rhs[-1]=budget
                scores=[]
                for label,objective in [('minimum',query),('maximum',-query)]:
                    result=linprog(np.r_[objective,np.zeros(n)],A_eq=augmented_eq,b_eq=rhs,A_ub=budget_matrix,b_ub=budget_rhs,method='highs')
                    if not result.success:
                        failures.append(dict(panel=panel,split=split,regime=regime,objective=label,error=result.message))
                        continue
                    flow=result.x[:n]
                    drift=np.max(np.abs(constraints@(flow-start)))
                    deviation=np.abs(flow-prior).sum()
                    assert drift<1e-5 and flow.min()>=-1e-8
                    assert deviation<=budget+1e-5
                    scores.append((float(query@flow/start.sum()),drift,deviation))
                if len(scores)==2:
                    low,high=scores
                    initial=query@start/start.sum()
                    assert low[0]<=high[0]+1e-9
                    rows.append(dict(panel=panel,split=split,regime=regime,threshold=int(threshold),
                        minimum_share=low[0],baseline_share=initial,maximum_share=high[0],width=high[0]-low[0],
                        max_constraint_drift=max(low[1],high[1]),min_endpoint_l1=low[2],
                        max_endpoint_l1=high[2],baseline_l1=l1_budget,minimum_l1=minimum.fun,budget=budget))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','regime']).width.agg(['min','mean','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
