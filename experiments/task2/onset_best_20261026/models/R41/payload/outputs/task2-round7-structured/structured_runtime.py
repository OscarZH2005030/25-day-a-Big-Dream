import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';sys.path[1:1]=[str(R6),str(R5)]
from ramp_runtime import RampPredictor
from special_features import features
from ramp_features import extras
from structured_model import StructuredQueue
from task2q.data import PANELS
class StructuredPredictor:
 def __init__(self,root=None,device='cpu'):
  torch.set_num_threads(4);self.root=Path(root or ROOT);self.device=device;f=json.loads((self.root/'frozen-calibration.json').read_text());self.rule=f['selected'];self.base=RampPredictor(R6);self.model=None
  for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
  assert hashlib.sha256((R6/'frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256']
  if self.rule['head']!='baseline':
   ck=torch.load(self.root/f"best-{self.rule['head']}.pt",map_location=device,weights_only=False);self.model=StructuredQueue(ck['mean'],ck['std']).to(device);self.model.load_state_dict(ck['model']);self.model.eval()
 @torch.inference_mode()
 def probability(self,s):
  b=self.base.probability(s)
  if self.model is None:return b
  n=s['x'].shape[1];r,p=extras(s);x=np.concatenate([features(s),r,p],axis=1)[:n];x=np.concatenate([x[:,:185],x[:,191:]],axis=1).astype('float32');q=self.model(torch.as_tensor(x,device=self.device),PANELS.index(s['panel']),s['condition'])[self.rule['head']].T.cpu().numpy();a=self.rule['alpha_neural'];return a*q+(1-a)*b
 def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype(np.int8)
