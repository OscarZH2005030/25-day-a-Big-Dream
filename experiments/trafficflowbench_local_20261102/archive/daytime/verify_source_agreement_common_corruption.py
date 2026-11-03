import hashlib
import numpy as np
import pandas as pd
import common as d
from task4_exact_pair_audit import base as inputs
from verify_cycle_budget import score

folder=d.OUT/'task4_source_agreement_common_corruption'
rows=pd.read_csv(folder/'by_case.csv',dtype={'seed':'int64'})
assert len(rows)==320 and rows.case.nunique()==80
checks=[]
for panel, group in rows.groupby('panel'):
    paths,A,*_=inputs.load(panel,'train')
    origin,dest=[paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone','destination_zone']]
    for case, table in group.groupby('case'):
        key=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
        casefile=d.WORK/'task4_source_agreement_common_corruption_cases'/key
        assert len(table)==4 and (table.artifact_sha256==d.sha(casefile)).all()
        with np.load(casefile) as z, np.load(d.WORK/'task4_source_agreement_stress_cases'/key) as ref:
            base,truth,count,priors=[ref[k] for k in ['baseline','truth','counts','priors']]
            b=priors[2]
            source=z['source_change']
            assert (priors[:2]+source).min()>=-1e-8
            assert np.abs(source).sum()<=.1*min(priors[:2].sum(axis=1))+1e-7
            assert int(z['seed'])==int(table.seed.iloc[0])
            baseline=score(base,truth,b,A,count,dest)[0]
            flows={}
            for row in table.itertuples():
                keyname=f'{row.version}_{row.rule}'
                flow=z[keyname]
                flows[keyname]=flow
                change=flow-base
                for delta in (source,change):
                    assert np.max(np.abs(A@delta))<1e-7
                    for idx in (origin,dest):
                        assert np.max(np.abs(np.bincount(idx,weights=delta)))<1e-7
                assert flow.min()>=-1e-8
                assert np.max(np.abs(change)-.05*base)<1e-7
                np.testing.assert_allclose(np.abs(flow-b).sum(),np.abs(base-b).sum(),atol=1e-7,rtol=0)
                np.testing.assert_allclose(np.abs(change).sum()/base.sum(),row.target_relative_movement,atol=1e-12,rtol=0)
                np.testing.assert_allclose(score(flow,truth,b,A,count,dest)[0]-baseline,row.od_delta,atol=1e-12,rtol=0)
            for rule in table.rule.unique():
                movement=np.abs(flows[f'clean_{rule}']-flows[f'perturbed_{rule}']).sum()/base.sum()
                checks.append(dict(case=case,noise=table.noise.iloc[0],rule=rule,prediction_movement=movement))
frame=pd.DataFrame(checks)
frame.groupby(['noise','rule']).prediction_movement.agg(['mean','median','max']).to_csv(folder/'prediction_sensitivity.csv')
print(frame.groupby(['noise','rule']).prediction_movement.mean().to_string())
