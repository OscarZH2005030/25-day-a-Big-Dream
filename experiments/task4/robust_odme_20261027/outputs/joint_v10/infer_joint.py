"""Reproduce frozen v10 exploratory policy without synthetic tuning."""
from pathlib import Path
import json
from run_joint import load,load_operator,REL,predict,np,pd
O=Path(__file__).resolve().parent
policy=json.loads((O/'policy.json').read_text(encoding='utf-8'))['policy']
frames=[]
for panel,name in policy.items():
 ids,ls,D,_=load_operator(REL/'corridors'/panel/'network')
 for split in ['validation','private']:
  A,c,b,t,_=load(panel,split,ids,ls,D);t=t.copy();t['path_flow']=predict(A,c,b)[name];frames.append(t)
out=pd.concat(frames,ignore_index=True);base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);keys=['panel','departure_time','path_id','origin_zone','destination_zone']
assert out[keys].astype(str).sort_values(keys).reset_index(drop=True).equals(base[keys].sort_values(keys).reset_index(drop=True))
assert np.isfinite(out.path_flow).all() and (out.path_flow>=0).all() and not out.duplicated(keys[:3]).any()
out.to_csv(O/'task4_joint_candidate.csv',index=False)
print('Frozen joint policy verified:',len(out),'rows')
