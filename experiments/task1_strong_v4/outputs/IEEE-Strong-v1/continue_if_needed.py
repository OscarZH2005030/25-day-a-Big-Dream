"""Validation-only continuation rule fixed before the primary holdout exists."""
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

root=Path(__file__).resolve().parent
workspace=root.parent.parent
primary=root/'training'
while not (primary/'accuracy-chart.png').exists():
    if (primary/'monitor-summary.json').exists():
        status=json.loads((primary/'monitor-summary.json').read_text())
        if status['status']!='completed':raise RuntimeError('Primary training did not finish')
    time.sleep(2)
history=json.loads((primary/'training-history.json').read_text())
evaluations=[v for v in history if 'development_score' in v]
checkpoint=torch.load(primary/'best-model.pt',map_location='cpu',weights_only=True)
best_epoch=checkpoint['epoch'];best_score=checkpoint['development_score']
improvement=evaluations[-1]['development_score']-evaluations[-3]['development_score']
refine=best_epoch>=14 and improvement>=.0008
decision=dict(rule='best epoch >=14 and last three internal scores improved >=0.0008',
              best_epoch=best_epoch,best_development_score=best_score,
              recent_development_improvement=improvement,refine=refine,holdout_score_read=False)
(root/'refinement-decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision),flush=True)
selected=primary
cache=workspace/'work/ieee-strong-cache-time8020'
reference=root/'references/transformer-all-metrics.csv'
if refine:
    folder=root/'refinement'
    folder.mkdir(exist_ok=True)
    with (folder/'launch.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(root/'train_refinement.py'),
            '--cache',str(cache),'--out',str(folder),'--reference',str(reference),
            '--init-checkpoint',str(primary/'best-model.pt'),'--epochs','8','--steps','60',
            '--accumulate','2','--lr','0.00005','--seed','20261021'],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (folder/'calibration.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',str(root/'calibrate_and_compare.py'),
            '--cache',str(cache),'--out',str(folder),'--reference',str(reference)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    refined=torch.load(folder/'best-model.pt',map_location='cpu',weights_only=True)
    if refined['development_score']>=best_score-1e-6:selected=folder
choice=dict(folder=selected.name,selection='Internal development only; no holdout-driven model choice',
            refinement_performed=refine)
(root/'selected-model.json').write_text(json.dumps(choice,indent=2))
subprocess.run([sys.executable,'-X','utf8',str(root/'write_result.py'),'--out',str(selected)],check=True)
(root/'experiment-complete.json').write_text(json.dumps(dict(status='completed',**choice),indent=2))
print('Accuracy experiment completed.',flush=True)
