"""Raw-observation adapter: frozen900-tree parent plus six residual heads."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
BASE900=Path(__file__).resolve().parents[2]/'best_overall_082639_jaccard_20261031'
sys.path.insert(0,str(BASE900/'code'))
from hybrid_ongoing import Hybrid as SharedHybrid,verify_manifest as verify_base,features,spatial_features,odds_mix

class Hybrid(SharedHybrid):
    def __init__(self,root):
        super().__init__(BASE900);self.root=Path(root);self.profile=json.loads((self.root/'profile.json').read_text());self.residuals={}
        for panel in self.models:
            self.residuals[panel]=[[lgb.Booster(model_file=str(self.root/'models'/f'final_{seed}'/f'{panel}-h{(h+1)*5}.txt')) for h in range(6)] for seed in self.profile['seeds']]
    def probability(self,s):
        if s['condition']==0:return self.onset.probability(s)
        import ramp_features as rf
        panel=s['panel'];basis=self.onset.bases[panel]['position'];base,names,_=features(s,rf.ramp_history(s),rf.topology(panel)['counts'],basis,self.context[panel]);extra,en=spatial_features(s,basis);assert names==self.names['base'] and en==self.names['extra'];x=np.column_stack([base,extra]).astype('float32');acc=np.zeros((len(x),6))
        for i,model in enumerate(self.models[panel]):
            for h in range(6):
                xx=np.column_stack([x,np.full(len(x),(h+1)*5)]);z=model.predict(xx,raw_score=True,num_threads=8)+self.profile['residual_alpha']*self.residuals[panel][i][h].predict(xx,raw_score=True,num_threads=8);acc[:,h]+=np.exp(-np.logaddexp(0.,-z))/len(self.models[panel])
        parent=np.asarray(self.parent.probability(s).T,dtype='float32');return odds_mix(parent,acc.astype('float32')).T

def verify_manifest(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    verify_base(BASE900);return len(manifest)
