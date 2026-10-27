"""Informed two-sided path adjustment under fixed link/destination constraints."""
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O.parent/'joint_v13'),str(O.parent/'joint_v14')]
from run_invariant import REL,load,load_operator,reference,invariant,cases,metric,folds,METRICS,TARGETS,np,pd
from run_guarded import gate
from run_bias import kl
from equality_projection import EqualProjector
SPECS={'v11':{'weight':0.,'positive_share':1.}}
for w in [.1,.25,.5,.75,1.]:
 SPECS[f'positive{w:g}']={'weight':w,'positive_share':1.}
 for share in [.25,.5,.75]:SPECS[f'balanced{share:g}_mix{w:g}']={'weight':w,'positive_share':share}
 SPECS[f'paired_gate_mix{w:g}']={'weight':w,'positive_share':'count_only_gate'}

def endpoints(A,c,b,paths,panel,projector):
 _,ref=reference(A,c,b,panel);q,st=kl(A,c,b,.05);assert st['success'];positive,e1=projector.project(q,ref);negative,e2=invariant(A,b,ref,paths.destination_zone.to_numpy());return ref,positive,negative,max(e1,e2)

def mix(name,A,c,b,ref,positive,negative):
 spec=SPECS[name];w=spec['weight'];a=spec['positive_share'];a=1-gate(A,c,b) if isinstance(a,str) else a
 return (1-w)*ref+w*(a*positive+(1-a)*negative)

def main():
 start=time.time();old=pd.read_csv(O.parent/'joint_v11/comparison_trials.csv');old=old[old.method=='joint_dev_v11'];oldcv=pd.read_csv(O.parent/'joint_v12/real_count_cv.csv');oldcv=oldcv[oldcv.candidate=='v11'];panels=sorted(p.name for p in (REL/'task4').iterdir());calrows=[];counts=[];comparison=[];checks=[];policy={};cert=[]
 for pn,p in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,base,_,_=load(p,'train',ids,ls,D);dest=paths.destination_zone.to_numpy();E=EqualProjector(A,dest);data=cases(pn,base,A,c,paths)
  for phase,scenario,f,b,cc in data:
   if phase!='calibration':continue
   ref,pos,neg,err=endpoints(A,cc,b,paths,p,E);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err))
   for name in SPECS:calrows.append(dict(panel=p,phase=phase,scenario=scenario,candidate=name,**metric(mix(name,A,cc,b,ref,pos,neg),f,b,A,cc,dest)))
  t=pd.DataFrame(calrows);t=t[t.panel==p];cal=t.groupby('candidate')[METRICS].mean();td=t[t.scenario.isin(TARGETS)].groupby(['scenario','candidate'])[METRICS].mean();eligible=['v11']
  for n in SPECS:
   d=cal.loc[n]-cal.loc['v11'];dd=td.xs(n,level='candidate')-td.xs('v11',level='candidate');ok=bool((d>=-1e-10).all() and (dd>=-1e-10).all().all());checks.append(dict(panel=p,candidate=n,synthetic_eligible=ok,target_worst_Sdev_delta=float(dd.S_dev.min()),**{'delta_'+k:float(d[k]) for k in METRICS}))
   if ok and n!='v11':eligible.append(n)
  for seed in [20261201,20261202,20261203,20261204]:
   fd=folds(A,seed)
   for k in range(3):
    if len(eligible)>1:
     M=A[fd!=k];cc=c[fd!=k];EE=EqualProjector(M,dest);ref,pos,neg,err=endpoints(M,cc,base,paths,p,EE)
     for n in eligible:
      pred=mix(n,M,cc,base,ref,pos,neg);counts.append(dict(panel=p,seed=seed,fold=k,candidate=n,NMAE=float(abs(A[fd==k]@pred-c[fd==k]).sum()/c[fd==k].sum())))
    else:
     v=oldcv[(oldcv.panel==p)&(oldcv.seed==seed)&(oldcv.fold==k)].iloc[0];counts.append(dict(panel=p,seed=seed,fold=k,candidate='v11',NMAE=float(v.NMAE)))
  ct=pd.DataFrame(counts);ct=ct[(ct.panel==p)&(ct.seed<=20261202)].groupby(['seed','candidate']).NMAE.mean().unstack();passed=[n for n in eligible if ct[n].mean()<=ct.v11.mean()*1.000001 and (ct[n]/ct.v11).max()<=1.01];score=td.groupby('candidate').S_dev.mean();chosen=str(score.loc[passed].idxmax());policy[p]=chosen
  for phase,scenario,f,b,cc in data:
   rr=old[(old.panel==p)&(old.phase==phase)&(old.scenario==scenario)].iloc[0];_,ref=reference(A,cc,b,p);baseline=metric(ref,f,b,A,cc,dest);assert max(abs(baseline[k]-rr[k]) for k in METRICS)<1e-10
   if chosen=='v11':m=baseline
   elif phase=='calibration':m=t[(t.scenario==scenario)&(t.candidate==chosen)].iloc[0][METRICS].to_dict()
   else:
    ref,pos,neg,err=endpoints(A,cc,b,paths,p,E);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err));m=metric(mix(chosen,A,cc,b,ref,pos,neg),f,b,A,cc,dest)
   comparison.extend([dict(panel=p,phase=phase,scenario=scenario,method='v11',**baseline),dict(panel=p,phase=phase,scenario=scenario,method='v15',**m)])
  pd.DataFrame(calrows).to_csv(O/'calibration_candidates.csv',index=False);pd.DataFrame(counts).to_csv(O/'real_count_cv.csv',index=False);pd.DataFrame(comparison).to_csv(O/'comparison_trials.csv',index=False);print('completed',p,chosen,flush=True)
 pd.DataFrame(checks).to_csv(O/'selection_checks.csv',index=False);pd.DataFrame(cert).to_csv(O/'invariance_certificates.csv',index=False);(O/'policy.json').write_text(json.dumps({'policy':policy,'specs':SPECS,'selection':'calibration all four plus total >=v11 overall and each target, count CV mean no worse/worst <=1.01, maximize target Sdev','selection_count_seeds':[20261201,20261202],'targets':TARGETS,'benchmark_previously_seen':True,'informed_direction':'raw KL path estimate projected to v11 counted-link and destination totals; never arbitrary nullspace inflation'},indent=2),encoding='utf-8')
 df=pd.DataFrame(comparison);df.groupby(['phase','method'])[METRICS].mean().to_csv(O/'four_components_overall.csv');df.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'four_components_by_scenario.csv');parts=[]
 for p,n in policy.items():
  for method,name in [('v11','v11'),('v15',n)]:
   q=pd.DataFrame(counts);q=q[(q.panel==p)&(q.candidate==name)].copy();q['method']=method;parts.append(q)
 pd.concat(parts).groupby(['seed','method']).NMAE.mean().to_csv(O/'real_count_overall.csv');(O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'same_fixed_cases':480,'candidate_count':len(SPECS),'max_constraint_error':max(r['max_constraint_error'] for r in cert),'CPU_threads':4,'hidden_truth_available':False,'baseline_preserved':True},indent=2),encoding='utf-8');print('POLICY',policy,flush=True);print(df.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
if __name__=='__main__':main()
