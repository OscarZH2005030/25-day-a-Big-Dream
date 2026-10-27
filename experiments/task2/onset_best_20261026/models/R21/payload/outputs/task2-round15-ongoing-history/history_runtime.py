import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
sys.path[1:1]=[str(OUT/n) for n in ['task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from seed_runtime import SeedPredictor
from coverage_data import matrix
from history_repair import extras
class HistoryPredictor:
    def __init__(self,root=None):
        self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.routes={r['panel']:r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.base=SeedPredictor(OUT/'task2-round13-seed-stability');self.models={}
        for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
        assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256'];assert hashlib.sha256((OUT/'task2-round13-seed-stability/frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256'];assert self.rule['thresholds']==self.base.rule['thresholds'];assert hashlib.sha256((OUT/'task2-round14-history-repair/history_repair.py').read_bytes()).hexdigest()==f['repair_source_sha256']
    def probability(self,s):
        b=self.base.probability(s)
        if s['condition']==0 or self.rule['alpha_history']==0:return b
        r=self.routes[s['panel']]
        if r['alpha_new']==0:return b
        if s['panel'] not in self.models:self.models[s['panel']]=lgb.Booster(model_file=str(self.root/(s['panel']+'-history.txt')))
        x=np.concatenate([matrix(s),extras(s)],axis=1);q=self.models[s['panel']].predict(x,num_threads=4).astype('float32').reshape(6,-1);adapt=r['alpha_new']*q+(1-r['alpha_new'])*b;a=self.rule['alpha_history'];return a*adapt+(1-a)*b
    def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype('int8')
