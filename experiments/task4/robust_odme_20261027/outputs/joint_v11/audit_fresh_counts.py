"""Freeze policy before two additional grouped count splits; train data reused."""
from run_joint_dev import O,REL,load,load_operator,predict,previous,folds,pd,json,hashlib
policy_bytes=(O/'policy.json').read_bytes();policy=json.loads(policy_bytes)['policy'];v10=json.loads((O.parent/'joint_v10/policy.json').read_text())['policy'];rows=[]
for panel,n in policy.items():
 ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,_,_=load(panel,'train',ids,ls,D)
 for seed in [20261205,20261206]:
  fd=folds(A,seed)
  for k in range(3):
   sols=predict(A[fd!=k],c[fd!=k],b,paths)
   for method,name in [('official','official'),('previous_v9',previous(panel)),('previous_v10',v10[panel]),('joint_dev_v11',n)]:rows.append(dict(panel=panel,seed=seed,fold=k,method=method,NMAE=float(abs(A[fd==k]@sols[name]-c[fd==k]).sum()/c[fd==k].sum())))
 print('fresh count audit',panel,flush=True)
assert (O/'policy.json').read_bytes()==policy_bytes
df=pd.DataFrame(rows);df.to_csv(O/'fresh_real_count_trials.csv',index=False);df.groupby(['seed','method']).NMAE.mean().to_csv(O/'fresh_real_count_overall.csv');(O/'fresh_audit_manifest.json').write_text(json.dumps({'policy_sha256':hashlib.sha256(policy_bytes).hexdigest(),'count_seeds':[20261205,20261206],'policy_not_changed':True,'data':'same released train counts, two new grouped partitions; not independent real path labels'},indent=2),encoding='utf-8');print(df.groupby(['seed','method']).NMAE.mean().to_string())
