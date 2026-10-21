"""Pre-planned per-corridor adaptation after global model selection."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch

root=Path(__file__).resolve().parent;workspace=root.parent.parent
def save(name,data):(root/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
if not (root/'specialist-policy.json').exists():
    save('specialist-policy.json',dict(recorded_at=dt.datetime.now().astimezone().isoformat(),
        revision=3,
        hybrid_holdout_exists=(root/'hybrid/training-result.json').exists(),
        earlier_public_holdouts_seen=True,reason='Test whether corridor-specific adaptation improves over global sharing',
        global_source='Internal development winner, selected without the global hybrid holdout score',
        rounds=2,steps_per_round=30,evaluate_every=1,base_learning_rate=.00005,
        attention_learning_rate=.00025,ema_decay=.9,gradient_accumulation=2,
        batch_probe=[1,2,4,8,16],seeds='20261023 + panel index',
        checkpoint_selection='Per-corridor internal development, with unmodified global initialization as epoch 0',
        calibration='Two scalars per corridor, internal dates only, bounds [0,2]',
        heldout_selection=False,
        graph_selection='Original versus distance-decayed proximity versus collapsed public-path relations, only for isolated receivers; choose on internal score with minimum gain 0.0001. Cached features keep the original graph.',
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ['train_specialist.py','calibrate_specialist.py','hybrid_model.py','quality_features.py','prepare_proximity_graphs.py','proximity-assets/manifest.json','prepare_path_graphs.py','path-assets/manifest.json']}))
save('specialist-progress.json',dict(phase='waiting_for_global'))
while not (root/'all-stages-complete.json').exists():time.sleep(2)
global_choice=json.loads((root/'selected-model.json').read_text())
source=root/global_choice['folder']/'best-model.pt'
cache=workspace/'work/ieee-strong-cache-64'
metadata=json.loads((cache/'metadata.json').read_text())
reference=root/'references/transformer-all-metrics.csv'
routes=[]
for index,pm in enumerate(metadata['panels']):
    panel=pm['panel'];out=root/'specialists'/panel;out.mkdir(parents=True,exist_ok=True)
    save('specialist-progress.json',dict(phase='training',panel=panel,index=index+1,total=10))
    if not (out/'training-result.json').exists():
        if (out/'best-model.pt').exists():raise RuntimeError('Partial expert exists; inspect it before resuming: '+str(out))
        with (out/'launch.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(root/'train_specialist.py'),
                '--cache',str(cache),'--out',str(out),'--reference',str(reference),
                '--init-checkpoint',str(source),'--panel',panel,'--epochs','2','--steps','30',
                '--eval-every','1','--accumulate','2','--lr','.00005','--seed',str(20261023+index)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (out/'calibrated-result.json').exists():
        with (out/'calibration.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-X','utf8',str(root/'calibrate_specialist.py'),
                '--cache',str(cache),'--out',str(out),'--reference',str(reference)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    checkpoint=torch.load(out/'best-model.pt',map_location='cpu',weights_only=True)
    routes.append(dict(panel=panel,checkpoint=str((out/'best-model.pt').relative_to(root)),
        calibration=str((out/'residual-calibration.json').relative_to(root)),
        model_graph=str((out/'model-graph.npz').relative_to(root)),
        graph_variant=checkpoint['config']['graph_variant'],
        best_epoch=checkpoint['epoch'],development_score=checkpoint['development_score']))
    print(json.dumps(dict(panel=panel,status='completed',best_epoch=checkpoint['epoch'])),flush=True)
save('final-model.json',dict(kind='corridor_specialists',global_source=global_choice['folder'],
    selection='All checkpoints and coefficients use internal development only',routes=routes))
subprocess.run([sys.executable,'-X','utf8',str(root/'write_specialist_result.py')],check=True)
save('specialist-progress.json',dict(phase='completed',total=10))
save('specialists-complete.json',dict(status='completed',model='final-model.json'))
print('All specialized models and the final report completed.',flush=True)
