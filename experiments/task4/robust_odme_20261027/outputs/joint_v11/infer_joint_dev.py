"""Inference from frozen policy; no hidden or artificial labels used."""
from run_joint_dev import O,REL,load,load_operator,predict,np,pd,json
policy=json.loads((O/'policy.json').read_text(encoding='utf-8'))['policy'];frames=[];rows=[]
for panel,name in policy.items():
 ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network')
 for split in ['validation','private']:
  A,c,b,t,_=load(panel,split,ids,ls,D);sols=predict(A,c,b,paths);t=t.copy();t['path_flow']=sols[name];frames.append(t)
  for method,n in [('official','official'),('v9','scaled_0.25' if panel=='D7_I210_W' else 'official'),('v11',name)]:rows.append(dict(panel=panel,split=split,method=method,S_od=None,S_link=max(0,1-float(abs(A@sols[n]-c).sum()/c.sum())),S_dev=None,S_attr=None,S_ODME=None,hidden_truth_available=False))
out=pd.concat(frames,ignore_index=True);base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);keys=['panel','departure_time','path_id','origin_zone','destination_zone'];assert out[keys].astype(str).sort_values(keys).reset_index(drop=True).equals(base[keys].sort_values(keys).reset_index(drop=True));assert np.isfinite(out.path_flow).all() and (out.path_flow>=0).all() and not out.duplicated(keys[:3]).any();out.to_csv(O/'task4_joint_dev_candidate.csv',index=False);pd.DataFrame(rows).to_csv(O/'actual_four_components.csv',index=False);print('Frozen v11 policy verified:',len(out),'rows')
