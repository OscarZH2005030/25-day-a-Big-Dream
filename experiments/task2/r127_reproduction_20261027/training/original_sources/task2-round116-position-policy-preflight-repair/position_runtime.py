
import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent
sys.path.insert(1,str(OUT/'task2-round41-paired-policy'))
from paired_runtime import PairedPredictor,normalized_mix
from position_mode_core import feature_matrix
from timing_core import unseen_queue

def reshape_to_parent(s,parent,raw,basis):
    if s['condition']!=0:return parent.copy()
    peak=parent.max(0);use=unseen_queue(s)&(peak>1e-6)&basis['supported'];v=raw[:,use];shape=v/np.maximum(v.max(0),1e-20);out=parent.copy();out[:,use]=shape*peak[use];out[v.argmax(0),np.flatnonzero(use)]=peak[use]
    assert np.array_equal(out.max(0),peak) and np.array_equal(out[:,~use],parent[:,~use]);return out

class PositionPredictor:
    def __init__(self,root=None):
        self.root=Path(root or ROOT);f=json.loads((self.root/'frozen-calibration.json').read_text(encoding='utf8'));self.rule=f['selected'];self.base=PairedPredictor(OUT/'task2-round41-paired-policy');self.routes={r['panel']:r for r in json.loads((self.root/'routes.json').read_text(encoding='utf8'))};self.models={};self.bases={};self.metadata={}
        assert self.rule['thresholds']==self.base.rule['thresholds']==[.5,.25]
        assert hashlib.sha256((OUT/'task2-round41-paired-policy/frozen-calibration.json').read_bytes()).hexdigest()==f['parent_frozen_sha256']
        assert hashlib.sha256((self.root/'routes.json').read_bytes()).hexdigest()==f['route_sha256']
        for n,h in f['hashes'].items():assert hashlib.sha256((self.root/n).read_bytes()).hexdigest()==h
    def probability(self,s):
        b=self.base.probability(s);p=s['panel'];alpha=self.rule['alpha_position']
        if s['condition']==1 or alpha==0 or not self.routes[p]['use_new']:return b
        if p not in self.models:
            self.models[p]=lgb.Booster(model_file=str(self.root/f'{p}-position.txt'))
            with np.load(self.root/f'{p}-basis.npz') as f:self.bases[p]={k:f[k] for k in ['position','random','supported']}
            self.metadata[p]=json.loads((self.root/f'{p}-basis.json').read_text(encoding='utf8'))
        assert list(s['network']['link_ids'])==self.metadata[p]['link_ids'];raw=self.models[p].predict(feature_matrix(s,self.bases[p],'position'),num_threads=4).astype('float32').reshape(6,-1);q=reshape_to_parent(s,b,raw,self.bases[p]);z=normalized_mix(s,b,q,alpha);assert np.array_equal(z.max(0),b.max(0)) and np.array_equal(z[:,~unseen_queue(s)],b[:,~unseen_queue(s)]);return z
    def predict(self,s):return (self.probability(s)>=self.rule['thresholds'][s['condition']]).astype('int8')
