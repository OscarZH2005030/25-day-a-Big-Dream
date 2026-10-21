"""Compare a warm-expanded attention/SSM hybrid using internal selection."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

root=Path(__file__).resolve().parent;workspace=root.parent.parent
def save(name,data):(root/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
save('extended-development-policy.json',dict(
    recorded_at=dt.datetime.now().astimezone().isoformat(),revision=3,
    primary_holdout_already_seen=True,refined_holdout_exists=(root/'refinement/training-result.json').exists(),
    supersedes='extended-development-policy-v1.json; the idle SSM-extension controller was stopped before it trained',
    reason='Internal refinement after two rounds improved only 0.000315; test content-dependent time retrieval instead of only extending the same architecture',
    model='Warm-expand shared graph SSM with two full-day temporal-attention adapters',
    features='64: preserve all 48 original channels and restore occupancy, availability and imputation inputs identified in the reference feature audit',
    trigger='After the existing eight-round SSM refinement finishes',
    maximum_extra_rounds=16,steps_per_round=60,base_learning_rate=.0001,
    attention_learning_rate=.0005,seed=20261022,
    checkpoint_selection='Internal development only; preserve the initial winner as epoch 0 and compare against the preceding candidate',
    caveat='Architecture and additional budget chosen after primary public holdout inspection. This is exploratory model development, not untouched-test evaluation.'))
cache=workspace/'work/ieee-strong-cache-64'
while not (root/'experiment-complete.json').exists() or not (cache/'metadata.json').exists():time.sleep(2)
choice=json.loads((root/'selected-model.json').read_text())
previous=root/choice['folder']
checkpoint=torch.load(previous/'best-model.pt',map_location='cpu',weights_only=True)
save('extended-development-decision.json',dict(variant='full_day_attention_ssm',previous=previous.name,
    best_epoch=checkpoint['epoch'],best_development_score=checkpoint['development_score'],
    decision_used_refined_holdout_metric=False))
out=root/'hybrid';out.mkdir(exist_ok=True)
save('extension-status.json',dict(status='training',folder=out.name))
reference=root/'references/transformer-all-metrics.csv'
with (out/'launch.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,'-X','utf8',str(root/'train_hybrid.py'),
        '--cache',str(cache),'--out',str(out),'--reference',str(reference),
        '--init-checkpoint',str(previous/'best-model.pt'),'--epochs','16','--steps','60',
        '--accumulate','2','--lr','.0001','--seed','20261022'],stdout=log,stderr=subprocess.STDOUT,check=True)
with (out/'calibration.log').open('w',encoding='utf-8') as log:
    subprocess.run([sys.executable,'-X','utf8',str(root/'calibrate_and_compare.py'),
        '--cache',str(cache),'--out',str(out),'--reference',str(reference)],stdout=log,stderr=subprocess.STDOUT,check=True)
candidate=torch.load(out/'best-model.pt',map_location='cpu',weights_only=True)
if candidate['development_score']>checkpoint['development_score']+1e-5:choice['folder']=out.name
choice.update(hybrid_performed=True,selection='Internal development only; public holdout does not choose checkpoints')
save('selected-model.json',choice)
subprocess.run([sys.executable,'-X','utf8',str(root/'write_result.py'),
                '--out',str(root/choice['folder'])],check=True)
save('extension-status.json',dict(status='completed',hybrid_performed=True))
save('all-stages-complete.json',dict(status='completed',**choice))
print('All bounded candidate training stages completed.',flush=True)
