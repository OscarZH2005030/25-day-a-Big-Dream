import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R12=ROOT.parent/'task2-round12-phase-soft';R11=ROOT.parent/'task2-round11-phase-validation';R10=ROOT.parent/'task2-round10-phase-coverage';R9=ROOT.parent/'task2-round9-hierarchical';R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';sys.path[1:1]=[str(R12),str(R11),str(R10),str(R9),str(R8),str(R7),str(R6),str(R5)]
from phase_soft_runtime import PhaseSoftPredictor
from special_features import features
from ramp_features import extras
SEEDS=[20261027,20261028,20261029]
class SeedPredictor:
 def __init__(self,root=None):
  self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.routes={r['panel']:r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.base=PhaseSoftPredictor(R12);self.models={}
  for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
  for n,h in f['seed0_hashes'].items():assert hashlib.sha256((R10/n).read_bytes()).hexdigest()==h
  assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256'];assert hashlib.sha256((R12/'frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256'];assert self.rule['thresholds'][1]==self.base.rule['thresholds'][1]
 def probability(self,s):
  b=self.base.probability(s)
  if s['condition']==1:return b
  r=self.routes[s['panel']];adapt=b
  if r['alpha_new']:
   rr,p=extras(s);x=np.concatenate([features(s),rr,p],axis=1);preds=[]
   for seed in SEEDS:
    key=(s['panel'],seed)
    if key not in self.models:
     path=R10/f"{s['panel']}-0-expanded.txt" if seed==SEEDS[0] else self.root/f"{s['panel']}-seed{seed}.txt";self.models[key]=lgb.Booster(model_file=str(path))
    preds.append(self.models[key].predict(x,num_threads=4).astype('float32').reshape(6,-1))
   q=np.mean(np.stack(preds),axis=0).astype('float32');adapt=r['alpha_new']*q+(1-r['alpha_new'])*b
  a=self.rule['alpha_seed'];return a*adapt+(1-a)*b
 def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype(np.int8)
