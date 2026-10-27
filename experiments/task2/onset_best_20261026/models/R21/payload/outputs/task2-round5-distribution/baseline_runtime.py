"""上一轮冻结预测的概率接口；Task1完全不在该运行时内。"""
import json,hashlib
from pathlib import Path
import lightgbm as lgb
from causal_features import enhanced
from task2q.data import PANELS
class Baseline:
 def __init__(self,root):
  self.root=Path(root);self.rule=json.loads((self.root/'frozen-calibration.json').read_text())['selected'];self.routes={(r['panel'],r['condition']):r for r in json.loads((self.root/'routes.json').read_text())};self.pool=lgb.Booster(model_file=str(self.root/'pooled.txt'));self.previous={(p,c):lgb.Booster(model_file=str(self.root/f'{p}-{c}-previous.txt')) for p in PANELS for c in [0,1]};self.adapted={k:lgb.Booster(model_file=str(self.root/r['model'])) for k,r in self.routes.items() if r['model']!='reference'}
 def probability(self,s):
  x=enhanced(s,s['network']);key=(s['panel'],s['condition']);po=self.pool.predict(x,num_threads=4).astype('float32');pr=self.previous[key].predict(x,num_threads=4).astype('float32');reference=.25*pr+.75*po;r=self.routes[key];adapt=reference if key not in self.adapted else r['alpha_to_pooled']*self.adapted[key].predict(x,num_threads=4).astype('float32')+(1-r['alpha_to_pooled'])*po;a=self.rule['alpha_adapted'];return (a*adapt+(1-a)*reference).reshape(6,-1)
