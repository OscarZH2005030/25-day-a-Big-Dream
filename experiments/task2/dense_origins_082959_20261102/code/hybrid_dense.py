"""Raw-observation adapter; unchanged onset, fixed candidate for every window."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[2]/'best_overall_082639_jaccard_20261031'
sys.path.insert(0,str(BASE/'code'))
from hybrid_ongoing import Hybrid as Parent,features,spatial_features,verify_manifest as verify_base
from dense_runtime import Portable
class Hybrid(Parent):
 def __init__(self,root,candidate='dense_100'):
  super().__init__(BASE);self.root=Path(root);self.profile=json.loads((self.root/'profile.json').read_text());self.candidate=candidate;assert candidate in self.profile['candidates'];self.runtime=Portable(self.root);self.runtime.preload(self.models.keys())
 def probability(self,s):
  if s['condition']==0:return self.onset.probability(s)
  import ramp_features as rf
  panel=s['panel'];basis=self.onset.bases[panel]['position'];x,n,_=features(s,rf.ramp_history(s),rf.topology(panel)['counts'],basis,self.context[panel]);extra,en=spatial_features(s,basis);assert n==self.names['base'] and en==self.names['extra'];x=np.column_stack([x,extra]).astype('float32');p=np.asarray(self.parent.probability(s).T,dtype='float32');return self.runtime.predict(panel,x,p,self.profile['candidates'][self.candidate]['alpha']).T
 def decode(self,s,p):
  return np.asarray(p)>=np.asarray(self.profile['candidates'][self.candidate]['thresholds'])[:,None] if s['condition']==1 else self.onset.decode(s,p)
def verify_manifest(root):
 root=Path(root)
 for p,h in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
 for p,h in json.loads((root/'dependency_sha256.json').read_text()).items():assert hashlib.sha256((root.parent/p).read_bytes()).hexdigest()==h,p
 verify_base(BASE)
