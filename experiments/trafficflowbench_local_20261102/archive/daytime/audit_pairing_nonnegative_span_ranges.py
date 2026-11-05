"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

import common as d
from audit_direct_od_measurement_design import base





def main():
    out=d.OUT/'task4_pairing_nonnegative_span_ranges'
    baseline=getattr(d, 'FROZEN_TASK4_BASELINE', None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        splits=['validation','private'],quantiles=[.25,.5,.75],
        method='Nonnegative LP bounds under baseline counts, both margins and exact prior L1 in baseline orthant. No draws.'))
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
            quantiles=span[order[np.searchsorted(np.cumsum(prior[order])/prior.sum(),[.25,.5,.75])]]
            sign=np.sign(start-prior)
            constraints=sparse.csr_matrix(np.vstack((counts,groups,sign)))
            rhs=constraints@start
            bounds=[(v,None) if s>0 else (0,v) if s<0 else (v,v) for v,s in zip(prior,sign)]
            for quantile,threshold in zip((.25,.5,.75),quantiles):
                query=(span>threshold).astype(float)
                scores=[]
                for label, objective in [('minimum',query),('maximum',-query)]:
                    result=linprog(objective,A_eq=constraints,b_eq=rhs,bounds=bounds,method='highs')
                    if not result.success:
                        failures.append(dict(panel=panel,split=split,quantile=quantile,objective=label,error=result.message))
                        continue
                    drift=np.max(np.abs(constraints@(result.x-start)))
                    l1_delta=abs(np.abs(result.x-prior).sum()-np.abs(start-prior).sum())
                    assert drift<1e-5 and l1_delta<1e-5 and result.x.min()>=-1e-8
                    scores.append((float(query@result.x/start.sum()),drift,l1_delta))
                if len(scores)==2:
                    low,high=scores
                    initial=query@start/start.sum()
                    assert low[0]-1e-9<=initial<=high[0]+1e-9
                    rows.append(dict(panel=panel,split=split,quantile=quantile,threshold=int(threshold),
                        minimum_share=low[0],baseline_share=initial,maximum_share=high[0],
                        width=high[0]-low[0],max_constraint_drift=max(low[1],high[1]),max_prior_l1_drift=max(low[2],high[2])))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','quantile']).width.agg(['min','mean','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
