"""Reduce unnecessary prior deviation while exactly preserving links/destinations."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,warnings
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O.parent/'joint_v12')]
from run_bias import REL,load,load_operator,reference,metric,folds,scenario_input,extra,SCENARIOS,EXTRA,METRICS,TARGETS
import numpy as np,pandas as pd
from scipy import sparse
from scipy.optimize import linprog,OptimizeWarning
WEIGHTS={'v11':0.,'invariant0.1':.1,'invariant0.25':.25,'invariant0.5':.5,'invariant1':1.}

def invariant(A,b,ref,dest):
 di=pd.factorize(dest)[0];n=len(b);B=sparse.csr_matrix((np.ones(n),(di,np.arange(n))),shape=(di.max()+1,n));M=sparse.vstack([A,B],format='csr');rhs=M@ref;I=sparse.eye(n,format='csr')
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',OptimizeWarning)
  r=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format='csr'),b_ub=np.r_[b,-b],A_eq=sparse.hstack([M,sparse.csr_matrix(M.shape)],format='csr'),b_eq=rhs,bounds=(0,None),method='highs',options={'threads':4,'time_limit':60.,'primal_feasibility_tolerance':1e-7,'dual_feasibility_tolerance':1e-7})
 assert r.success,r.message
 x=np.maximum(r.x[:n],0);err=float(abs(M@x-rhs).max());assert err<1e-5,(err,r.message);assert abs(x-b).sum()<=abs(ref-b).sum()+1e-5
 return x,err

def cases(pn,base,A,c,paths):
 out=[]
 for phase,offset,sigma in [('calibration',0,.3),('evaluation',10000,.8),('sparse_stress',20000,.8)]:
  rng=np.random.default_rng(101000+pn+offset);f0=base*c.sum()/(A@base).sum()*rng.lognormal(-.5*sigma*sigma,sigma,len(base))
  if phase=='sparse_stress':f0*=np.where(rng.random(len(base))<.15,6.4,.05)
  for sn,scenario in enumerate(SCENARIOS+EXTRA):
   rng=np.random.default_rng(102000+pn*100+sn+offset);f,b,cc=(scenario_input if scenario in SCENARIOS else extra)(f0.copy(),A,paths,scenario,rng);out.append((phase,scenario,f,b,cc))
 return out

def main():
 start=time.time();panels=sorted(p.name for p in (REL/'task4').iterdir());old=pd.read_csv(O.parent/'joint_v11/comparison_trials.csv');old=old[old.method=='joint_dev_v11'];oldcv=pd.read_csv(O.parent/'joint_v12/real_count_cv.csv');oldcv=oldcv[oldcv.candidate=='v11'];calrecords=[];cv=[];comparisons=[];checks=[];policy={};cert=[]
 for pn,p in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,base,_,_=load(p,'train',ids,ls,D);dest=paths.destination_zone.to_numpy();cache=cases(pn,base,A,c,paths)
  for phase,scenario,f,b,cc in cache:
   if phase!='calibration':continue
   _,ref=reference(A,cc,b,p);z,err=invariant(A,b,ref,dest);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err))
   for name,w in WEIGHTS.items():
    x=(1-w)*ref+w*z;calrecords.append(dict(panel=p,phase=phase,scenario=scenario,candidate=name,**metric(x,f,b,A,cc,dest)))
  t=pd.DataFrame(calrecords);t=t[t.panel==p];cal=t.groupby('candidate')[METRICS].mean();td=t[t.scenario.isin(TARGETS)].groupby(['scenario','candidate'])[METRICS].mean();eligible=['v11']
  for n in WEIGHTS:
   d=cal.loc[n]-cal.loc['v11'];dd=td.xs(n,level='candidate')-td.xs('v11',level='candidate');ok=bool((d>=-1e-10).all() and (dd>=-1e-10).all().all());checks.append(dict(panel=p,candidate=n,synthetic_eligible=ok,**{'delta_'+k:float(d[k]) for k in METRICS},target_worst_Sdev_delta=float(dd.S_dev.min())))
   if ok and n!='v11':eligible.append(n)
  # Do not spend LP solves on candidates that already failed four-component guards.
  for seed in [20261201,20261202,20261203,20261204]:
   fd=folds(A,seed)
   for k in range(3):
    if len(eligible)>1:
     _,ref=reference(A[fd!=k],c[fd!=k],base,p);z,err=invariant(A[fd!=k],base,ref,dest)
     for n in eligible:
      w=WEIGHTS[n];x=(1-w)*ref+w*z;cv.append(dict(panel=p,seed=seed,fold=k,candidate=n,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())))
    else:
     v=oldcv[(oldcv.panel==p)&(oldcv.seed==seed)&(oldcv.fold==k)].iloc[0];cv.append(dict(panel=p,seed=seed,fold=k,candidate='v11',NMAE=float(v.NMAE)))
  ct=pd.DataFrame(cv);ct=ct[(ct.panel==p)&(ct.seed<=20261202)].groupby(['seed','candidate']).NMAE.mean().unstack();passed=[]
  for n in eligible:
   if ct[n].mean()<=ct.v11.mean()*1.000001 and (ct[n]/ct.v11).max()<=1.01:passed.append(n)
  score=td.groupby('candidate').S_dev.mean();chosen=str(score.loc[passed].idxmax());policy[p]=chosen
  for phase,scenario,f,b,cc in cache:
   rr=old[(old.panel==p)&(old.phase==phase)&(old.scenario==scenario)].iloc[0];_,ref=reference(A,cc,b,p);baseline=metric(ref,f,b,A,cc,dest);assert max(abs(baseline[k]-rr[k]) for k in METRICS)<1e-10
   if chosen=='v11':m=baseline
   elif phase=='calibration':m=t[(t.scenario==scenario)&(t.candidate==chosen)].iloc[0][METRICS].to_dict()
   else:
    z,err=invariant(A,b,ref,dest);cert.append(dict(panel=p,phase=phase,scenario=scenario,max_constraint_error=err));w=WEIGHTS[chosen];m=metric((1-w)*ref+w*z,f,b,A,cc,dest)
   comparisons.extend([dict(panel=p,phase=phase,scenario=scenario,method='v11',**baseline),dict(panel=p,phase=phase,scenario=scenario,method='v13',**m)])
  pd.DataFrame(calrecords).to_csv(O/'calibration_candidates.csv',index=False);pd.DataFrame(cv).to_csv(O/'real_count_cv.csv',index=False);pd.DataFrame(comparisons).to_csv(O/'comparison_trials.csv',index=False);print('completed',p,chosen,flush=True)
 pd.DataFrame(checks).to_csv(O/'selection_checks.csv',index=False);pd.DataFrame(cert).to_csv(O/'invariance_certificates.csv',index=False);(O/'policy.json').write_text(json.dumps({'policy':policy,'weights':WEIGHTS,'selection':'all-four and total guards overall and each target on calibration; count CV mean no worse and worst seed <=1.01; maximize target Sdev','targets':TARGETS,'selection_count_seeds':[20261201,20261202],'old_benchmark_seen':True},indent=2),encoding='utf-8')
 df=pd.DataFrame(comparisons);df.groupby(['phase','method'])[METRICS].mean().to_csv(O/'four_components_overall.csv');df.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'four_components_by_scenario.csv');parts=[]
 for p,n in policy.items():
  for method,name in [('v11','v11'),('v13',n)]:
   q=pd.DataFrame(cv);q=q[(q.panel==p)&(q.candidate==name)].copy();q['method']=method;parts.append(q)
 pd.concat(parts).groupby(['seed','method']).NMAE.mean().to_csv(O/'real_count_overall.csv');(O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'same_fixed_cases':480,'weights':WEIGHTS,'invariant_constraints':'all published counted link loads and normalized AND absolute destination totals equal v11','max_constraint_error':max(r['max_constraint_error'] for r in cert),'CPU_threads':4,'hidden_truth_available':False,'baseline_preserved':True},indent=2),encoding='utf-8');print(df.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
if __name__=='__main__':main()
