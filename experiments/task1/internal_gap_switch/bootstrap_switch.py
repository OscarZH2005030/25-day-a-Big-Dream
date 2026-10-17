"""Paired calendar-day resampling of frozen internal predictions, no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).parent/'results';plan=json.loads((root/'plan.json').read_text())
if json.loads((root/'status.json').read_text())['state']!='complete':raise SystemExit('Inference not complete')
dates=plan['dates'];panels=plan['panels'];methods=['lgb','tf','hybrid'];regimes=['R1','R2','R3']
cube=np.zeros((len(dates),len(panels),3,7));rows=[]
for pi,panel in enumerate(panels):
    frame=pd.read_parquet(root/panel/'internal_predictions.parquet')
    day=frame.timestamp.astype(str).str[:10]
    values=pd.DataFrame({'date':day,'regime':frame.mask_regime,'n':1})
    for m in methods:
        values[m+'_speed_sse']=(frame['speed_'+m]-frame.speed_truth)**2
        values[m+'_flow_sse']=(frame['flow_per_lane_'+m]-frame.flow_per_lane_truth)**2
    cols=['n']+[m+'_'+c+'_sse' for m in methods for c in ['speed','flow']]
    sums=values.groupby(['date','regime'],as_index=False)[cols].sum()
    sums['panel']=panel;rows.append(sums)
    for row in sums.itertuples(index=False):
        cube[dates.index(row.date),pi,regimes.index(row.regime)]=[getattr(row,c) for c in cols]
pd.concat(rows,ignore_index=True).to_csv(root/'daily_errors.csv',index=False)

def scores(total):
    n=total[...,0];sv=[];sq=[]
    for i in range(3):
        sv.append(np.maximum(0,1-np.sqrt(total[...,1+2*i]/n)/25))
        sq.append(np.maximum(0,1-np.sqrt(total[...,2+2*i]/n)/600))
    return {'LightGBM':(.54*sv[0]+.46*sq[0]).mean(axis=(-1,-2)),
            'Transformer':(.54*sv[1]+.46*sq[1]).mean(axis=(-1,-2)),
            'hybrid_both':(.54*sv[2]+.46*sq[2]).mean(axis=(-1,-2)),
            'hybrid_speed_only':(.54*sv[2]+.46*sq[0]).mean(axis=(-1,-2)),
            'hybrid_flow_only':(.54*sv[0]+.46*sq[2]).mean(axis=(-1,-2))}

observed=scores(cube.sum(axis=0))
reference=pd.read_csv(root/'method_summary.csv').set_index('method').S_state
for key,value in observed.items():assert abs(float(value)-reference[key])<1e-12
rng=np.random.default_rng(2026);valid=[]
while len(valid)<1000:
    picks=rng.integers(len(dates),size=(100,len(dates)))
    for pick in picks:
        total=cube[pick].sum(axis=0)
        if (total[...,0]>0).all():valid.append(total)
        if len(valid)==1000:break
samples=scores(np.stack(valid));result=[]
for name in ['hybrid_both','hybrid_speed_only','hybrid_flow_only']:
    for baseline in ['LightGBM','Transformer']:
        delta=samples[name]-samples[baseline]
        result.append(dict(method=name,baseline=baseline,observed_delta=float(observed[name]-observed[baseline]),
                            ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975)),
                            replicates=1000,unit='shared calendar day across all corridors'))
out=pd.DataFrame(result);out.to_csv(root/'paired_calendar_bootstrap.csv',index=False)
print(out.to_string(index=False))
