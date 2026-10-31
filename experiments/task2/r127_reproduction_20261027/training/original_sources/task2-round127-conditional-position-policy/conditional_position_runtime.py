
import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
sys.path.insert(1,str(OUT/'task2-round116-position-policy-preflight-repair'))
from position_runtime import PositionPredictor
from paired_runtime import normalized_mix
from timing_core import features,unseen_queue
from position_mode_core import extras
from gated_core import concentrated_adjust

def head_features(s,basis):
    x=features(s);z=np.column_stack([x,extras(s,basis,'position')]).astype('float32');assert z.shape==(len(x),916);return z

def compose(s,b,anchor,old,new):
    q=concentrated_adjust(s,b,anchor,new,1.);keep=(old.max(1)<.5)|(~unseen_queue(s));q[:,keep]=b[:,keep];assert np.array_equal(q.max(0),b.max(0)) and np.array_equal(q[:,keep],b[:,keep]);return q

class ConditionalPositionPredictor:
    def __init__(self,root=None):
        self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.base=PositionPredictor(OUT/'task2-round116-position-policy-preflight-repair');self.routes={r['panel']:r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.bases={};self.metadata={};self.model=None;self.old=None
        assert self.rule['thresholds']==self.base.rule['thresholds']==[.5,.25]
        assert hashlib.sha256((OUT/'task2-round116-position-policy-preflight-repair/frozen-calibration.json').read_bytes()).hexdigest()==f['parent_frozen_sha256']
        assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256']
        for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
    def probability(self,s):
        b=self.base.probability(s);p=s['panel'];alpha=self.rule['alpha_position']
        if s['condition']==1 or alpha==0 or not self.routes[p]['use_new']:return b
        if self.model is None:self.model=lgb.Booster(model_file=str(self.root/'position-first-full.txt'));self.old=lgb.Booster(model_file=str(self.root/'old-first-full.txt'))
        if p not in self.bases:
            with np.load(self.root/(p+'-basis.npz')) as f:self.bases[p]={k:f[k] for k in ['position','random','supported']}
            self.metadata[p]=json.loads((self.root/(p+'-basis.json')).read_text(encoding='utf8'))
        assert list(s['network']['link_ids'])==self.metadata[p]['link_ids'];anchor=self.base.base.anchor.probability(s);assert np.array_equal(anchor.max(0),b.max(0));old=self.old.predict(features(s),num_threads=4).astype('float32');new=self.model.predict(head_features(s,self.bases[p]),num_threads=4).astype('float32');q=compose(s,b,anchor,old,new);z=normalized_mix(s,b,q,alpha);assert np.array_equal(z.max(0),b.max(0)) and np.array_equal(z[:,~unseen_queue(s)],b[:,~unseen_queue(s)]);return z
    def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype('int8')
