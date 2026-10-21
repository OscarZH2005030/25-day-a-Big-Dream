"""Resume CPU reporting after all ten inference/metric computations passed.

The first report failed because the archived reference lacks family labels.
No model selection, predictions, or training are changed here.
"""
import json
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from optimize import ROOT,V1,CACHE,save,sha

metadata=json.loads((CACHE/'metadata.json').read_text())
routes={r['panel']:r for r in json.loads((ROOT/'frozen-routes.json').read_text())['routes']}
local=json.loads((ROOT/'final-model.json').read_text())['local_flow_routes']
current=pd.read_csv(ROOT/'holdout-by-regime.csv')
previous=pd.read_csv(ROOT/'previous-holdout-by-regime.csv')
reference=pd.read_csv(V1/'references/transformer-all-metrics.csv')
reference=reference[reference.regime.isin(['R1','R2','R3'])].copy()
reference['family']=reference.panel.map({p['panel']:p['family'] for p in metadata['panels']})
daily=pd.read_csv(ROOT/'paired-daily-errors.csv')
by=pd.read_csv(ROOT/'comparison-by-panel.csv')
assert len(current)==len(previous)==len(reference)==30
assert len(daily)==550
assert set(current.panel)==set(routes)
logs=(ROOT/'evaluate.log').read_text(encoding='utf-8')
assert logs.count('"phase": "holdout_evaluated"')==10
assert 'KeyError' in logs and "KeyError: 'family'" in logs
matched=previous.merge(pd.read_csv(V1/'specialists/calibrated-holdout-by-regime.csv'),on=['panel','regime'],suffixes=('_recomputed','_archived'),validate='one_to_one')
assert np.array_equal(matched.n_recomputed,matched.n_archived)
assert np.max(np.abs(matched.S_state_recomputed-matched.S_state_archived))<2e-8
def summarize(frame):
    return dict(score=float(frame.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean()),
        speed_rmse=float(np.sqrt(np.dot(frame.n,frame.rmse_speed_kmh**2)/frame.n.sum())),
        flow_rmse=float(np.sqrt(np.dot(frame.n,frame.rmse_flow_per_lane**2)/frame.n.sum())))
result=dict(status='evaluated',previous=summarize(previous),new=summarize(current),transformer=summarize(reference),
    targets=int(current.n.sum()),directions_improved_vs_previous=int((by.delta_vs_previous>0).sum()),
    directions_improved_vs_transformer=int((by.delta_vs_transformer>0).sum()),
    adapted_routes=sum(bool(v['checkpoint']) for v in routes.values()),local_flow_routes=sum(r['alpha']>0 for r in local),
    selection='Frozen internal-only selections; this evaluation does not choose or change routes.',
    limitation='Exploratory reuse of a historically observed holdout. Not hidden-test evidence.')
assert result['targets']==6636705
save(ROOT/'evaluation-summary.json',result)
save(ROOT/'checkpoint-inference-parity.json',dict(passed=True,prediction_tolerance=2e-6,roads=10,
    evidence='All ten holdout_evaluated events occur after their checkpoint/development parity assertions in evaluate.py; numerical per-road maximum differences were not persisted before reporting failed. No GPU checks repeated.'))
arr=np.zeros((55,10,3,5));panels=list(routes)
for row in daily.itertuples():
    arr[row.day-218,panels.index(row.panel),['R1','R2','R3'].index(row.regime)]=[row.n,row.speed_sq,row.flow_sq,row.old_speed_sq,row.old_flow_sq]
rng=np.random.default_rng(20261024);differences=[];flow_differences=[]
for _ in range(2000):
    s=arr[rng.integers(55,size=55)].sum(0);counts=s[...,0]
    if (counts==0).any():continue
    a=np.sqrt(s[...,1:3]/counts[...,None]);b=np.sqrt(s[...,3:5]/counts[...,None])
    differences.append(float((np.maximum(0,1-a)*[.54,.46]).sum(-1).mean()-(np.maximum(0,1-b)*[.54,.46]).sum(-1).mean()))
    flow_differences.append(float((np.sqrt(s[...,2].sum()/counts.sum())-np.sqrt(s[...,4].sum()/counts.sum()))*600))
save(ROOT/'paired-bootstrap.json',dict(replicates=len(differences),score_delta_95pct=np.quantile(differences,[.025,.975]).tolist(),
    flow_rmse_delta_95pct=np.quantile(flow_differences,[.025,.975]).tolist(),
    limitation='Descriptive paired date-block bootstrap conditional on selected models. Cannot remove selection bias from historical holdout reuse.'))
preserved=json.loads((ROOT/'preserved-v1-sha256.json').read_text())
assert all(sha(Path(p))==v for p,v in preserved.items())
save(ROOT/'preservation-check.json',dict(passed=True,files=len(preserved)))
manifest=json.loads((ROOT/'final-model.json').read_text())
save(ROOT/'reporting-repair.json',dict(recorded_at=dt.datetime.now().astimezone().isoformat(),
    failure='Archived Transformer reference has no family column. Map it using the fixed panel metadata.',
    original_evaluate_source_sha256=manifest['source_sha256']['evaluate.py'],
    repaired_evaluate_source_sha256=sha(ROOT/'evaluate.py'),
    resume_source_sha256=sha(Path(__file__)),predictions_changed=False,models_changed=False,selections_changed=False,
    gpu_repeated=False,note='CPU-only completion from all 30 saved group metrics and 550 saved daily errors.'))
print(json.dumps(result,indent=2))
