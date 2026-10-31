"""Fixed best-onset / unchanged R127-ongoing inference adapter."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from onset_interval_core import softmax,interval_distribution,PATTERNS,interval_decode
from onset_context_core import PastContext
from two_stage_onset import causal_precursors

class Hybrid:
    def __init__(self,root):
        self.root=Path(root);self.profile=json.loads((self.root/'profile.json').read_text());self.models=[];self.context={};self.bases={}
        # Load new heads before R127 installs its archive-only loader guard.
        for seed in self.profile['seeds']:
            self.models.append(tuple(lgb.Booster(model_file=str(p)) for p in [self.root/f'shared/final_{seed}_event.txt',self.root/f'shared/final_{seed}_timing.txt',self.root/f'final_early_{seed}/model.txt']))
        for file in (self.root/'basis').glob('*.npz'):
            panel=file.stem.removesuffix('-basis')
            with np.load(file) as z:self.bases[panel]={k:z[k] for k in ['position','random','supported']}
            self.context[panel]=PastContext.load(self.root/'final_reference'/f'{panel}.npz')
        self.parent=None
    def attach_parent(self,parent):self.parent=parent
    def features(self,s):
        from timing_core import features
        from position_mode_core import extras
        import ramp_features as rf
        precursor,_=causal_precursors(s);base=np.column_stack([features(s),extras(s,self.bases[s['panel']],'position'),precursor,np.isfinite(precursor)]).astype('float32');assert base.shape[1]==940
        p,c,_,_=self.context[s['panel']].features(s,rf.ramp_history(s),rf.topology(s['panel'])['counts']);x=np.column_stack([base,p,c]).astype('float32');assert x.shape[1]==996;return x
    def probability(self,s):
        if s['condition']==1:
            if self.parent is None:raise RuntimeError('Attach frozen R127 parent first')
            return self.parent.probability(s)
        x=self.features(s);ps=[]
        for event,timing,early in self.models:
            a=softmax(timing.predict(np.column_stack([x[:,:940],np.zeros(len(x))]),raw_score=True,num_threads=8));b=softmax(timing.predict(np.column_stack([x[:,:940],np.ones(len(x))]),raw_score=True,num_threads=8));q=interval_distribution(a,b)@PATTERNS
            for h in range(3):q[:,h]=early.predict(np.column_stack([x,np.full(len(x),h)]),num_threads=8)
            e=np.exp(-np.logaddexp(0.,-event.predict(x,raw_score=True,num_threads=8)));ps.append((e[:,None]*q).astype('float32'))
        return np.mean(ps,0).T
    def decode(self,s,p):
        if s['condition']==1:return np.asarray(p)>=self.profile['ongoing_threshold']
        return interval_decode(np.asarray(p).T,np.array(self.profile['onset_threshold'])).T

def verify_manifest(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,value in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,f'Changed published file: {name}'
    return len(manifest)
