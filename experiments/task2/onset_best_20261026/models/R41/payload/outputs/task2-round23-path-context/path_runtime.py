import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
sys.path[1:1]=[str(OUT/n) for n in ['task2-round21-link-category','task2-round19-interval-budget','task2-round18-interval-likelihood','task2-round17-onset-hazard','task2-round15-ongoing-history','task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from category_runtime import CategoryPredictor
from path_data import features
class PathPredictor:
    def __init__(self,root=None):
        self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.routes={(r['panel'],r['condition']):r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.graphs=json.loads((self.root/'path-graphs.json').read_text(encoding='utf8'));self.base=CategoryPredictor(OUT/'task2-round21-link-category');self.models={}
        for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
        assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256'];assert hashlib.sha256((self.root/'path-graphs.json').read_bytes()).hexdigest()==f['graph_sha256'];assert hashlib.sha256((self.root/'link-vocabulary.json').read_bytes()).hexdigest()==f['vocabulary_sha256'];assert hashlib.sha256((self.root/'path_data.py').read_bytes()).hexdigest()==f['source_sha256'];assert hashlib.sha256((OUT/'task2-round21-link-category/frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256'];assert self.rule['thresholds']==self.base.rule['thresholds']
    def probability(self,s):
        b=self.base.probability(s);key=(s['panel'],s['condition']);a=self.rule['alphas_path'][s['condition']];r=self.routes[key]
        if a==0 or r['alpha_new']==0:return b
        if key not in self.models:self.models[key]=lgb.Booster(model_file=str(self.root/f'{key[0]}-{key[1]}-path.txt'))
        q=self.models[key].predict(features(s,self.graphs[s['panel']]),num_threads=4).astype('float32').reshape(6,-1);adapt=r['alpha_new']*q+(1-r['alpha_new'])*b;return a*adapt+(1-a)*b
    def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype('int8')
