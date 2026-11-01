"""Frozen best onset plus road-specialist ongoing, using only causal observations."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
ONSET=Path(__file__).resolve().parents[2]/'best_overall_081796_20261030'
sys.path.insert(0,str(ONSET/'code'))
from hybrid import Hybrid as FrozenOnset
from onset_context_core import PastContext
from ongoing_core import features
from speed_distribution_core import spatial_features,odds_mix

class Hybrid:
    def __init__(self,root):
        self.root=Path(root);self.profile=json.loads((self.root/'profile.json').read_text());self.names=json.loads((self.root/'feature_names.json').read_text());self.onset=FrozenOnset(ONSET);self.models={};self.context={};self.parent=None
        # All new LightGBM heads load before the unchanged R127 guard is installed.
        for file in (self.root/'final_reference').glob('*.npz'):
            panel=file.stem;self.context[panel]=PastContext.load(file);self.models[panel]=[lgb.Booster(model_file=str(self.root/'models'/f'final_{seed}'/f'{panel}.txt')) for seed in self.profile['seeds']]
    def attach_parent(self,parent):self.parent=parent;self.onset.attach_parent(parent)
    def probability(self,s):
        if s['condition']==0:return self.onset.probability(s)
        import ramp_features as rf
        panel=s['panel'];basis=self.onset.bases[panel]['position'];base,names,_=features(s,rf.ramp_history(s),rf.topology(panel)['counts'],basis,self.context[panel]);extra,en=spatial_features(s,basis);assert names==self.names['base'] and en==self.names['extra'];x=np.column_stack([base,extra]).astype('float32');acc=np.zeros((len(x),6))
        for model in self.models[panel]:
            for h in range(6):acc[:,h]+=np.exp(-np.logaddexp(0.,-model.predict(np.column_stack([x,np.full(len(x),(h+1)*5)]),raw_score=True,num_threads=8)))/len(self.models[panel])
        parent=np.asarray(self.parent.probability(s).T,dtype='float32');return odds_mix(parent,acc.astype('float32')).T
    def decode(self,s,p):
        return np.asarray(p)>=self.profile['ongoing_threshold'] if s['condition']==1 else self.onset.decode(s,p)

def verify_manifest(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,value in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,f'Changed published file: {name}'
    # The dependency's own manifest verifies every unchanged onset file.
    from hybrid import verify_manifest as verify_onset
    verify_onset(ONSET)
    return len(manifest)
