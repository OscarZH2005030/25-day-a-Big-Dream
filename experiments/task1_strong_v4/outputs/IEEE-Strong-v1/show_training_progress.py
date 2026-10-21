"""Compact read-only status for the currently active specialist."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text()) if path.exists() else None
state=read(root/'specialist-progress.json') or {}
answer=dict(state)
answer['completed_directions']=len(list((root/'specialists').glob('*/calibrated-result.json')))
if 'panel' in state:
    folder=root/'specialists'/state['panel']
    live=read(folder/'progress.json')
    if live:answer['current']={k:v for k,v in live.items() if k in ['phase','epoch','epochs','step','steps','seconds','mean_loss','best_development_score']}
    graph=read(folder/'graph-selection.json')
    if graph:answer['graph']=graph
    batch=read(folder/'batch-calibration.json')
    if batch:
        answer['micro_batch']=batch['selected']
        answer['gradient_accumulation']=batch['gradient_accumulation']
        answer['effective_batch']=batch['selected']*batch['gradient_accumulation']
    history=read(folder/'training-history.json')
    if history:
        points=[v for v in history if 'development_score' in v]
        if points:answer['latest_internal']=points[-1]['development_score']
print(json.dumps(answer,indent=2))
