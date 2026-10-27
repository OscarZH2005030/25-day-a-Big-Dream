"""第六轮冻结推理；保留第五轮作为明确基线依赖。"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R5=ROOT.parent/'task2-round5-distribution';sys.path.insert(1,str(R5))
from distribution_runtime import DistributionPredictor
from special_features import features
from ramp_features import extras
class RampPredictor:
 def __init__(self,root=None):
  self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text());self.rule=f['selected'];self.routes={(r['panel'],r['condition']):r for r in json.loads((self.root/'routes.json').read_text())};self.baseline=DistributionPredictor(R5);self.models={}
  for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
  assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256'];assert hashlib.sha256((R5/'frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256']
 def probability(self,s):
  b=self.baseline.probability(s);r=self.routes[(s['panel'],s['condition'])];adapt=b
  if r['variant']!='baseline':
   name=f"{s['panel']}-{s['condition']}-{r['variant']}"
   if name not in self.models:self.models[name]=lgb.Booster(model_file=str(self.root/(name+'.txt')))
   rr,ph=extras(s);x=np.concatenate([features(s),rr]+([ph] if r['variant']=='physics' else []),axis=1);raw=self.models[name].predict(x,num_iteration=r['iteration'],num_threads=4).astype('float32').reshape(6,-1);adapt=r['alpha_new']*raw+(1-r['alpha_new'])*b
  a=self.rule['alpha_ramp'];return a*adapt+(1-a)*b
 def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype(np.int8)
