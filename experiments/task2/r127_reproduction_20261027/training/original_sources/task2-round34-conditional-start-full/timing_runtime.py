import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
sys.path[1:1]=[str(OUT/n) for n in ['task2-round31-negative-window','task2-round30-custom-iou-control','task2-round27-earlier-onset','task2-round25-overlap-weights','task2-round23-path-context','task2-round21-link-category','task2-round19-interval-budget','task2-round18-interval-likelihood','task2-round17-onset-hazard','task2-round15-ongoing-history','task2-round14-history-repair','task2-round13-seed-stability','task2-round12-phase-soft','task2-round11-phase-validation','task2-round10-phase-coverage','task2-round9-hierarchical','task2-round8-soft-targets','task2-round7-structured','task2-round6-ramp-physics','task2-round5-distribution','task2-round2']]
from balanced_runtime import BalancedPredictor
from timing_core import features,unseen_queue,adjust
def normalized_mix(s,base,adapt,alpha):
    if alpha==0:return base.copy()
    peak=base.max(0);v=(1-alpha)*base+alpha*adapt;v*=peak/np.maximum(v.max(0),1e-8);v=np.minimum(v,peak);v[np.argmax(v,axis=0),np.arange(v.shape[1])]=peak
    keep=~unseen_queue(s);v[:,keep]=base[:,keep]
    assert np.array_equal(v.max(0),base.max(0)) and np.array_equal(v[:,keep],base[:,keep]);return v
class TimingPredictor:
    def __init__(self,root=None):
        self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.routes={r['panel']:r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.base=BalancedPredictor(OUT/'task2-round31-negative-window');self.model=None
        for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
        assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256'];assert hashlib.sha256((OUT/'task2-round31-negative-window/frozen-calibration.json').read_bytes()).hexdigest()==f['baseline_frozen_sha256'];assert self.rule['thresholds']==self.base.rule['thresholds']
    def probability(self,s):
        b=self.base.probability(s)
        if s['condition']==1 or self.rule['alpha_timing']==0 or self.routes[s['panel']]['alpha_new']==0:return b
        if self.model is None:self.model=lgb.Booster(model_file=str(self.root/'conditional-first-full.txt'))
        p=self.model.predict(features(s),num_threads=4).astype('float32');q=adjust(s,b,p,self.routes[s['panel']]['alpha_new']);return normalized_mix(s,b,q,self.rule['alpha_timing'])
    def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype('int8')
