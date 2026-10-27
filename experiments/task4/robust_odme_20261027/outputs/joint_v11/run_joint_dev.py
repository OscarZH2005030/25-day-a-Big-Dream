"""Fixed v10 benchmark; deviation-aware and origin/destination structured candidates."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O.parent/'joint_v10'),str(O.parent/'robust_v9'),str(O.parent/'improvement_v2'),str(O.parent)]
from run_joint import REL,load,load_operator,solve,kl,metric,folds,scenario_input,extra,SCENARIOS,EXTRA,METRICS
import numpy as np,pandas as pd

def projected(A,c,b):
 x,st=solve(A,c,b,'ridge',.05)
 if not st['success']:
  # Near convergence, large dual objective cancellation can stall Armijo.
  # Refine the same convex objective using residual-norm backtracking.
  from scipy.linalg import solve as linear_solve
  Q=A.toarray();y=A@x-c
  def fg(y):
   z=np.maximum(b-Q.T@y/.05,0);return y+c-Q@z,z
  for _ in range(80):
   g,x=fg(y)
   if abs(g).max()<1e-6:break
   H=(Q*(x>0)/.05)@Q.T+np.eye(len(c));step=linear_solve(H,-g,assume_a='pos');rate=1.
   for _ in range(40):
    if np.linalg.norm(fg(y+rate*step)[0])<np.linalg.norm(g):break
    rate*=.5
   y+=rate*step
  g,x=fg(y);assert abs(g).max()<1e-5,st
 return x

def predict(A,c,b,paths):
 x=projected(A,c,b);q,st=kl(A,c,b,.05);assert st['success'];s,st=kl(A,c,b*c.sum()/max(float((A@b).sum()),1e-12),.05);assert st['success']
 out={'official':x,'raw_1':q,'scaled_1':s,'scaled_0.25':.75*x+.25*s,'scaled_0.1':.9*x+.1*s}
 for label,y,amps in [('raw',q,[1.1,1.25,1.5]),('scaled',s,[1.1,1.25])]:
  for a in amps:
   z=projected(A,c,np.maximum(b+a*(y-b),1e-12));out[f'{label}_amp{a:g}']=z
   for w in [.25,.5]:out[f'{label}_amp{a:g}_mix{w:g}']=(1-w)*x+w*z
 # Smooth learned log path corrections by shared origin/destination before projection.
 # This uses observed counts only; no artificial truth or scenario label is passed here.
 oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0];r=np.log(np.maximum(q,1e-12)/np.maximum(b,1e-12));w=np.maximum(b,1e-9)
 def aggregate(idx):return np.bincount(idx,weights=w*r)/np.maximum(np.bincount(idx,weights=w),1e-12)
 ro=aggregate(oi)[oi];rd=aggregate(di)[di];mu=float(np.average(r,weights=w))
 for label,corr in [('origin',ro),('destination',rd),('origin_destination',ro+rd-mu)]:
  z=projected(A,c,b*np.exp(np.clip(corr,-5,5)));out['structured_'+label]=z;out['structured_'+label+'_mix0.5']=.5*(x+z)
 return out

def previous(panel):return 'scaled_0.25' if panel=='D7_I210_W' else 'official'

def main():
 start=time.time();records=[];cv=[];hashes=[];cache={};panels=sorted(p.name for p in (REL/'task4').iterdir());old=pd.read_csv(O.parent/'joint_v10/trials.csv')
 for pn,p in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,base,t,_=load(p,'train',ids,ls,D);cache[p]=(ids,ls,D,paths)
  for phase,offset,sigma in [('calibration',0,.3),('evaluation',10000,.8),('sparse_stress',20000,.8)]:
   rng=np.random.default_rng(101000+pn+offset);f0=base*c.sum()/(A@base).sum()*rng.lognormal(-.5*sigma*sigma,sigma,len(base))
   if phase=='sparse_stress':f0*=np.where(rng.random(len(base))<.15,6.4,.05)
   for sn,scenario in enumerate(SCENARIOS+EXTRA):
    rng=np.random.default_rng(102000+pn*100+sn+offset);f,b,cc=(scenario_input if scenario in SCENARIOS else extra)(f0.copy(),A,paths,scenario,rng)
    hashes.append(dict(panel=p,phase=phase,scenario=scenario,sha256=hashlib.sha256(f.tobytes()+b.tobytes()+cc.tobytes()).hexdigest()))
    sols=predict(A,cc,b,paths)
    for name,x in sols.items():
     m=metric(x,f,b,A,cc,paths.destination_zone.to_numpy())
     if name=='official':
      ref=old[(old.panel==p)&(old.phase==phase)&(old.scenario==scenario)&(old.candidate=='official')].iloc[0]
      assert max(abs(m[k]-ref[k]) for k in METRICS)<1e-10,'Fixed benchmark mismatch'
     records.append(dict(panel=p,phase=phase,scenario=scenario,candidate=name,**m,deviation_ratio=float(abs(x-b).sum()/max(abs(f-b).sum(),1e-12))))
  for seed in [20261201,20261202,20261203,20261204]:
   fd=folds(A,seed)
   for k in range(3):
    for name,x in predict(A[fd!=k],c[fd!=k],base,paths).items():cv.append(dict(panel=p,seed=seed,fold=k,candidate=name,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())))
  pd.DataFrame(records).to_csv(O/'trials.csv',index=False);pd.DataFrame(cv).to_csv(O/'real_count_cv.csv',index=False);print('completed',p,flush=True)
 tr=pd.DataFrame(records);counts=pd.DataFrame(cv);policy={};checks=[]
 for p in panels:
  refname=previous(p);cal=tr[(tr.panel==p)&(tr.phase=='calibration')].groupby('candidate')[METRICS].mean();ref=cal.loc[refname]
  cc=counts[(counts.panel==p)&(counts.seed<=20261202)].groupby(['seed','candidate']).NMAE.mean().unstack();sc=tr[(tr.panel==p)&(tr.phase=='calibration')].groupby(['scenario','candidate']).synthetic_S_ODME.mean().unstack();eligible=[refname]
  for n in cal.index:
   d=cal.loc[n]-ref;ratio=cc[n]/cc[refname];meanratio=cc[n].mean()/cc[refname].mean();worstscenario=float((sc[n]-sc[refname]).min())
   ok=bool((d[METRICS]>=-1e-10).all() and meanratio<=1.000001 and ratio.max()<=1.01 and worstscenario>=-.005)
   checks.append(dict(panel=p,candidate=n,reference=refname,eligible=ok,count_mean_ratio=meanratio,count_worst_ratio=float(ratio.max()),worst_scenario_total_delta=worstscenario,**{'delta_'+k:float(d[k]) for k in METRICS}))
   if ok:eligible.append(n)
  policy[p]=str(cal.loc[list(dict.fromkeys(eligible))].sort_values(['S_dev','synthetic_S_ODME'],ascending=False).index[0])
 pd.DataFrame(checks).to_csv(O/'selection_checks.csv',index=False);pd.DataFrame(hashes).to_csv(O/'fixed_case_hashes.csv',index=False)
 (O/'policy.json').write_text(json.dumps({'policy':policy,'selection':'maximize calibration Sdev subject to all four components and total >= previous v9, actual count CV mean no worse and worst seed <=1.01; worst scenario total loss<=.005','evaluation_not_used_in_selection':True,'same_v10_cases':True,'repeated_benchmark_seen':True},indent=2),encoding='utf-8')
 parts=[];cparts=[];v10=json.loads((O.parent/'joint_v10/policy.json').read_text())['policy']
 for method,pol in [('official',{p:'official' for p in panels}),('previous_v9',{p:previous(p) for p in panels}),('previous_v10',v10),('joint_dev_v11',policy)]:
  for p,n in pol.items():
   q=tr[(tr.panel==p)&(tr.candidate==n)].copy();q['method']=method;parts.append(q)
   q=counts[(counts.panel==p)&(counts.candidate==n)].copy();q['method']=method;cparts.append(q)
 joined=pd.concat(parts);joined.to_csv(O/'comparison_trials.csv',index=False);joined.groupby(['phase','method'])[METRICS].mean().to_csv(O/'four_components_overall.csv');joined.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'four_components_by_scenario.csv');pd.concat(cparts).to_csv(O/'comparison_count_trials.csv',index=False);pd.concat(cparts).groupby(['seed','method']).NMAE.mean().to_csv(O/'real_count_overall.csv');tr.groupby(['phase','candidate'])[METRICS].mean().to_csv(O/'all_candidates_overall.csv')
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'synthetic_cases':len(hashes),'same_case_baseline_max_difference_tolerance':1e-10,'candidates':tr.candidate.nunique(),'CPU_threads':4,'hidden_path_truth':False,'baseline_preserved':True},indent=2),encoding='utf-8');print('POLICY',policy,flush=True);print(joined.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
if __name__=='__main__':main()
