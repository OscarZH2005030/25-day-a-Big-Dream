"""Freeze policy, infer full templates and audit two new real count partitions."""
from run_invariant import O,REL,load,load_operator,reference,invariant,folds,np,pd,json,WEIGHTS
policy_bytes=(O/'policy.json').read_bytes();policy=json.loads(policy_bytes)['policy'];frames=[];actual=[];rows=[];cert=[]
def predict(A,c,b,panel,dest,name):
 _,ref=reference(A,c,b,panel);w=WEIGHTS[name]
 if not w:return ref,ref,0.
 z,err=invariant(A,b,ref,dest);return ref,(1-w)*ref+w*z,err
for panel,name in policy.items():
 ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network');dest=paths.destination_zone.to_numpy()
 for split in ['validation','private']:
  A,c,b,t,_=load(panel,split,ids,ls,D);ref,z,err=predict(A,c,b,panel,dest,name);t=t.copy();t['path_flow']=z;frames.append(t);cert.append(dict(panel=panel,split=split,max_constraint_error=err))
  for method,x in [('v11',ref),('v13',z)]:actual.append(dict(panel=panel,split=split,method=method,S_od=None,S_link=max(0,1-float(abs(A@x-c).sum()/c.sum())),S_dev=None,S_attr=None,S_ODME=None,hidden_path_truth_available=False))
 A,c,b,_,_=load(panel,'train',ids,ls,D)
 for seed in [20261207,20261208]:
  fd=folds(A,seed)
  for k in range(3):
   ref,z,err=predict(A[fd!=k],c[fd!=k],b,panel,dest,name)
   for method,x in [('v11',ref),('v13',z)]:rows.append(dict(panel=panel,seed=seed,fold=k,method=method,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())))
 print('inference and fresh count audit',panel,flush=True)
assert (O/'policy.json').read_bytes()==policy_bytes
out=pd.concat(frames,ignore_index=True);base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);keys=['panel','departure_time','path_id','origin_zone','destination_zone'];assert out[keys].astype(str).sort_values(keys).reset_index(drop=True).equals(base[keys].sort_values(keys).reset_index(drop=True));assert np.isfinite(out.path_flow).all() and (out.path_flow>=0).all() and not out.duplicated(keys[:3]).any();out.to_csv(O/'task4_invariant_candidate.csv',index=False);pd.DataFrame(actual).to_csv(O/'actual_four_components.csv',index=False);df=pd.DataFrame(rows);df.to_csv(O/'fresh_count_trials.csv',index=False);df.groupby(['seed','method']).NMAE.mean().to_csv(O/'fresh_count_overall.csv');pd.DataFrame(cert).to_csv(O/'actual_invariance_certificates.csv',index=False);print('Verified',len(out),'rows')
