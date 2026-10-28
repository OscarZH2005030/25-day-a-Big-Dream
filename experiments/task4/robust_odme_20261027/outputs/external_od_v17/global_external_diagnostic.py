"""Freeze a global calibration-mean candidate, disclose regressions; no promotion."""
from run_external_od import O,REL,load,load_operator,reference,endpoints,mix,EqualProjector,cases,metric,folds,METRICS,TARGETS,SPECS,np,pd,json
import hashlib
cal=pd.read_csv(O/'calibration_candidates.csv').groupby('candidate')[METRICS].mean();delta=cal-cal.loc['v11'];eligible=delta[(delta>=-1e-10).all(axis=1)].index;score=.5*(delta.S_od+delta.S_dev);name=str(score.loc[eligible].idxmax());source=SPECS[name]['source'];selection={'chosen':name,'selection':'global calibration-mean Sod+Sdev maximum, guarding all four means and total; deliberately diagnostic, NOT the strict per-panel/target policy','target_family_guards_not_enforced_in_selection':True,'benchmark_previously_seen':True,'promoted':False};(O/'global_diagnostic_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8');frozen=(O/'global_diagnostic_selection.json').read_bytes();trials=[];counts=[];cert=[]
for pn,p in enumerate(sorted(p.name for p in (REL/'task4').iterdir())):
 ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,b,_,_=load(p,'train',ids,ls,D);dest=paths.destination_zone.to_numpy();E=EqualProjector(A,dest)
 for phase,scenario,f,bb,cc in cases(pn,b,A,c,paths):
  ref,ends,err=endpoints(A,cc,bb,paths,p,E,sources=[source] if source else []);pred=mix(name,ref,ends);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err))
  for method,x in [('v11',ref),('global_external_diagnostic',pred)]:trials.append(dict(panel=p,phase=phase,scenario=scenario,method=method,**metric(x,f,bb,A,cc,dest)))
 for seed in [20261201,20261202,20261203,20261204,20261211,20261212]:
  fd=folds(A,seed)
  for k in range(3):
   M=A[fd!=k];cc=c[fd!=k];EE=EqualProjector(M,dest);ref,ends,err=endpoints(M,cc,b,paths,p,EE,sources=[source] if source else []);pred=mix(name,ref,ends)
   for method,x in [('v11',ref),('global_external_diagnostic',pred)]:counts.append(dict(panel=p,seed=seed,fold=k,method=method,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum()),count_type='released_synthetic'))
 print('global diagnostic',p,name,flush=True)
assert (O/'global_diagnostic_selection.json').read_bytes()==frozen
df=pd.DataFrame(trials);df.to_csv(O/'global_diagnostic_trials.csv',index=False);df.groupby(['phase','method'])[METRICS].mean().to_csv(O/'global_diagnostic_four_components.csv');df.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'global_diagnostic_by_scenario.csv');cc=pd.DataFrame(counts);cc.to_csv(O/'global_diagnostic_released_count_trials.csv',index=False);cc.groupby(['seed','method']).NMAE.mean().to_csv(O/'global_diagnostic_released_count_overall.csv');pd.DataFrame(cert).to_csv(O/'global_diagnostic_invariance.csv',index=False);print(df.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
