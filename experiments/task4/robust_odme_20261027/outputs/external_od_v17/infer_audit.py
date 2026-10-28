"""Frozen external-prior inference; audit released synthetic counts explicitly."""
from run_external_od import O,REL,load,load_operator,reference,endpoints,mix,EqualProjector,folds,np,pd,json
raw=(O/'policy.json').read_bytes();policy=json.loads(raw)['policy'];frames=[];actual=[];rows=[]
def predict(A,c,b,paths,panel,name):
 if name=='v11':
  _,ref=reference(A,c,b,panel);return ref,ref
 ref,ends,err=endpoints(A,c,b,paths,panel,EqualProjector(A,paths.destination_zone.to_numpy()));return ref,mix(name,ref,ends)
for panel,name in policy.items():
 ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network')
 for split in ['validation','private']:
  A,c,b,t,_=load(panel,split,ids,ls,D);ref,z=predict(A,c,b,paths,panel,name);t=t.copy();t['path_flow']=z;frames.append(t)
  for method,x in [('v11',ref),('external_od_v17',z)]:actual.append(dict(panel=panel,split=split,method=method,S_od=None,S_link=max(0,1-float(abs(A@x-c).sum()/c.sum())),S_dev=None,S_attr=None,S_ODME=None,count_type='released_synthetic',hidden_path_truth_available=False))
 A,c,b,_,_=load(panel,'train',ids,ls,D)
 for seed in [20261211,20261212]:
  fd=folds(A,seed)
  for k in range(3):
   ref,z=predict(A[fd!=k],c[fd!=k],b,paths,panel,name)
   for method,x in [('v11',ref),('external_od_v17',z)]:rows.append(dict(panel=panel,seed=seed,fold=k,method=method,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum()),count_type='released_synthetic'))
 print('frozen inference and released synthetic count audit',panel,flush=True)
assert (O/'policy.json').read_bytes()==raw
out=pd.concat(frames,ignore_index=True);base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);keys=['panel','departure_time','path_id','origin_zone','destination_zone'];assert out[keys].astype(str).sort_values(keys).reset_index(drop=True).equals(base[keys].sort_values(keys).reset_index(drop=True));assert np.isfinite(out.path_flow).all() and (out.path_flow>=0).all() and not out.duplicated(keys[:3]).any();out.to_csv(O/'task4_external_od_candidate.csv',index=False);pd.DataFrame(actual).to_csv(O/'released_four_components.csv',index=False);df=pd.DataFrame(rows);df.to_csv(O/'fresh_released_count_trials.csv',index=False);df.groupby(['seed','method']).NMAE.mean().to_csv(O/'fresh_released_count_overall.csv');print('Verified',len(out),'rows')
