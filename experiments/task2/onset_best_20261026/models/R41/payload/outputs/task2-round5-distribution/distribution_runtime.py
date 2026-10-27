"""分布驱动模型冻结推理；仅加载当前目录中的模型与因果特征。"""
import json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from baseline_runtime import Baseline
from special_features import features,latest_state
class DistributionPredictor:
 def __init__(self,root=None):
  self.root=Path(root or Path(__file__).resolve().parent);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.routes={(r['panel'],r['condition']):r for r in json.loads((self.root/'routes.json').read_text())};self.base=Baseline(self.root/'baseline');self.models={}
  for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h,'Frozen model changed: '+n
  assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256']
 def model(self,name):
  if name not in self.models:self.models[name]=lgb.Booster(model_file=str(self.root/(name+'.txt')))
  return self.models[name]
 def raw(self,s,config):
  xx=features(s);n=s['x'].shape[1];tag=f"{s['panel']}-{s['condition']}";kind=config['kind'];iteration=config.get('iteration')
  if kind=='direct':return self.model(tag+'-direct').predict(xx,num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1)
  if kind=='event_factor':
   ex=np.concatenate([xx[:n,:185],xx[:n,191:]],axis=1);pe=self.model(tag+'-event').predict(ex,num_threads=4);pc=self.model(tag+'-conditional').predict(xx,num_threads=4).reshape(6,-1);return (pe[None]*pc).astype('float32')
  if kind=='transition':
   pf=self.model(tag+'-transition').predict(xx,num_iteration=iteration,num_threads=4).reshape(6,-1);_,queued=latest_state(s);return np.where(queued[None],1-pf,pf).astype('float32')
  if kind=='speed_residual':
   residual=self.model(tag+'-speed').predict(xx,num_threads=4).reshape(6,-1);last,_=latest_state(s);future=last[None]+residual;cut=s['network']['threshold']/s['network']['free_speed'];return (1/(1+np.exp(np.clip((future-cut[None])/config['temperature'],-30,30)))).astype('float32')
  raise ValueError(kind)
 def probability(self,s):
  b=self.base.probability(s);r=self.routes[(s['panel'],s['condition'])];adapt=b if r['config']['kind']=='baseline' else r['alpha_new']*self.raw(s,r['config'])+(1-r['alpha_new'])*b;a=self.rule['alpha_distribution'];return a*adapt+(1-a)*b
 def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype(np.int8)
