import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R11=ROOT.parent/'task2-round11-phase-validation';R10=ROOT.parent/'task2-round10-phase-coverage';R9=ROOT.parent/'task2-round9-hierarchical';R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';sys.path[1:1]=[str(R11),str(R10),str(R9),str(R8),str(R7),str(R6),str(R5)]
from phase_runtime import PhasePredictor
from special_features import features
from ramp_features import extras
class PhaseSoftPredictor:
 def __init__(self,root=None):
  self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.routes={r['panel']:r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.base=PhasePredictor(R11);self.models={}
  for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
  assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256'];assert hashlib.sha256((R11/'frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256'];assert self.rule['thresholds'][1]==self.base.rule['thresholds'][1]
 def probability(self,s):
  b=self.base.probability(s)
  if s['condition']==1:return b
  r=self.routes[s['panel']];adapt=b
  if r['sigma'] is not None:
   name=f"{s['panel']}-sigma{r['sigma']}"
   if name not in self.models:self.models[name]=lgb.Booster(model_file=str(self.root/(name+'.txt')))
   rr,p=extras(s);x=np.concatenate([features(s),rr,p],axis=1);q=self.models[name].predict(x,num_threads=4).astype('float32').reshape(6,-1);adapt=r['alpha_new']*q+(1-r['alpha_new'])*b
  a=self.rule['alpha_phase_soft'];return a*adapt+(1-a)*b
 def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype(np.int8)
