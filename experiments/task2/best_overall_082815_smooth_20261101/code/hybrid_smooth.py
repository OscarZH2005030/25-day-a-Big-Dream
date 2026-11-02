"""Frozen model with January-selected horizon thresholds; onset unchanged."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
FROZEN=Path(__file__).resolve().parents[2]/'best_overall_082723_horizon_20261031'
sys.path.insert(0,str(FROZEN/'code'))
from hybrid_horizon import Hybrid as FrozenHybrid,verify_manifest as verify_frozen
class Hybrid(FrozenHybrid):
    def __init__(self,root):
        super().__init__(FROZEN);self.root=Path(root);self.profile=json.loads((self.root/'profile.json').read_text())
    def decode(self,s,p):
        return np.asarray(p)>=np.asarray(self.profile['ongoing_threshold'])[:,None] if s['condition']==1 else self.onset.decode(s,p)
def verify_manifest(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    verify_frozen(FROZEN);return len(manifest)
