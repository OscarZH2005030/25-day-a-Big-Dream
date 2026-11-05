"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

import common as d
from audit_direct_od_measurement_design import base





def main():
    out=d.OUT/'task4_pairing_minimum_l1_histogram'
    baseline=getattr(d, 'FROZEN_TASK4_BASELINE', None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        splits=['validation','private'],regimes=['margins','margins_and_histogram'],
        method='Compare the top-half prior OD query on unrestricted-sign minimum-L1 faces with and without the frozen baseline span histogram. No draws.'))
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
            n=len(paths)
            bins=(span[:,None]==np.unique(span)).astype(float).T
            query=np.zeros(n)
            query[np.argsort(-prior,kind='stable')[:n//2]]=1.
            identity=sparse.eye(n,format='csr')
            deviations=sparse.vstack((sparse.hstack((identity,-identity)),sparse.hstack((-identity,-identity))),format='csr')
            deviation_rhs=np.r_[prior,-prior]
            l1_objective=np.r_[np.zeros(n),np.ones(n)]
            budget_matrix=sparse.vstack((deviations,sparse.csr_matrix(l1_objective[None,:])),format='csr')
            for regime in ('margins','margins_and_histogram'):
                components=[counts,groups]
                if regime=='margins_and_histogram':
                    components.append(bins)
                constraints=sparse.csr_matrix(np.vstack(components))
                rhs=constraints@start
                equalities=sparse.hstack((constraints,sparse.csr_matrix(constraints.shape)),format='csr')
                minimum=linprog(l1_objective,A_eq=equalities,b_eq=rhs,A_ub=deviations,b_ub=deviation_rhs,method='highs')
                if not minimum.success:
                    failures.append(dict(panel=panel,split=split,regime=regime,objective='l1',error=minimum.message))
                    continue
                budget=minimum.fun+1e-8*start.sum()
                scores=[]
                for label, objective in [('minimum',query),('maximum',-query)]:
                    result=linprog(np.r_[objective,np.zeros(n)],A_eq=equalities,b_eq=rhs,
                                   A_ub=budget_matrix,b_ub=np.r_[deviation_rhs,budget],method='highs')
                    if not result.success:
                        failures.append(dict(panel=panel,split=split,regime=regime,objective=label,error=result.message))
                        continue
                    flow=result.x[:n]
                    drift=np.max(np.abs(constraints@(flow-start)))
                    assert drift<1e-5 and np.abs(flow-prior).sum()<=budget+1e-5 and flow.min()>=-1e-8
                    scores.append((float(query@flow/start.sum()),drift))
                if len(scores)==2:
                    low,high=scores
                    assert high[0]>=low[0]-1e-9
                    rows.append(dict(panel=panel,split=split,regime=regime,minimum_share=low[0],
                        baseline_share=query@start/start.sum(),maximum_share=high[0],width=high[0]-low[0],
                        minimum_l1=minimum.fun,baseline_l1=np.abs(start-prior).sum(),
                        max_constraint_drift=max(low[1],high[1])))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','regime']).width.agg(['min','mean','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
