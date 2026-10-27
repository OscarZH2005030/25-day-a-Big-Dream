"""Fixed benchmark: sparse prior adjustments with original-count reprojection."""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O.parent/'joint_v11'),str(O.parent/'joint_v10')]
from run_joint_dev import REL,load,load_operator,projected,kl,metric,folds,scenario_input,extra,SCENARIOS,EXTRA,METRICS
import numpy as np,pandas as pd
from scipy.linalg import solve as lin_solve
TARGETS=['localized_undercount','count_systematic30']

def elastic(A,c,b,strength):
 Q=A.toarray();lam=.05;eta=lam*np.median(b)*strength;y=np.zeros(len(c))
 def fg(y):
  v=Q.T@y;u=-v;d=np.sign(u)*np.maximum(abs(u)-eta,0)/lam;x=np.maximum(b+d,0)
  val=-v@x-.5*lam*np.dot(x-b,x-b)-eta*abs(x-b).sum()+c@y+.5*y@y
  return val,y+c-Q@x,x,((abs(u)>eta)&(x>0))
 for it in range(300):
  val,g,x,active=fg(y)
  if abs(g).max()<1e-6:break
  H=(Q*active/lam)@Q.T+np.eye(len(c));step=lin_solve(H,-g,assume_a='pos');rate=1.
  for _ in range(55):
   vv,gg,_,_=fg(y+rate*step)
   if vv<=val+1e-4*rate*(g@step)+1e-8 or (np.linalg.norm(g)<.01 and np.linalg.norm(gg)<np.linalg.norm(g)):break
   rate*=.5
  y+=rate*step
 _,g,x,_=fg(y);assert abs(g).max()<1e-5,(it,float(abs(g).max()),strength)
 return x

def reference(A,c,b,panel):
 x=projected(A,c,b);ref=x
 if panel in ['D7_I210_W','D7_I405_S']:
  s,st=kl(A,c,b*c.sum()/max(float((A@b).sum()),1e-12),.05);assert st['success']
  if panel=='D7_I210_W':ref=.75*x+.25*projected(A,c,np.maximum(b+1.1*(s-b),1e-12))
  else:ref=.9*x+.1*s
 return x,ref

def chosen(A,c,b,panel,name):
 x,ref=reference(A,c,b,panel)
 if name=='v11':return ref
 if name=='official':return x
 label,w=name.split('_mix');a=float(label[len('elastic'):]);w=float(w);z=projected(A,c,elastic(A,c,b,a));return (1-w)*ref+w*z

def predict(A,c,b,panel):
 x,ref=reference(A,c,b,panel);out={'v11':ref,'official':x}
 for a in [.25,1.,4.]:
  # Always score against original counts; final projection does not replace observations.
  z=projected(A,c,elastic(A,c,b,a))
  for w in [.1,.25,.5,1.]:out[f'elastic{a:g}_mix{w:g}']=(1-w)*ref+w*z
 return out

def main():
 start=time.time();records=[];cv=[];panels=sorted(p.name for p in (REL/'task4').iterdir());old=pd.read_csv(O.parent/'joint_v11/comparison_trials.csv')
 for pn,p in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,base,_,_=load(p,'train',ids,ls,D)
  for phase,offset,sigma in [('calibration',0,.3),('evaluation',10000,.8),('sparse_stress',20000,.8)]:
   rng=np.random.default_rng(101000+pn+offset);f0=base*c.sum()/(A@base).sum()*rng.lognormal(-.5*sigma*sigma,sigma,len(base))
   if phase=='sparse_stress':f0*=np.where(rng.random(len(base))<.15,6.4,.05)
   for sn,scenario in enumerate(SCENARIOS+EXTRA):
    rng=np.random.default_rng(102000+pn*100+sn+offset);f,b,cc=(scenario_input if scenario in SCENARIOS else extra)(f0.copy(),A,paths,scenario,rng)
    for name,x in predict(A,cc,b,p).items():
     m=metric(x,f,b,A,cc,paths.destination_zone.to_numpy())
     if name=='v11':
      ref=old[(old.panel==p)&(old.phase==phase)&(old.scenario==scenario)&(old.method=='joint_dev_v11')].iloc[0];assert max(abs(m[k]-ref[k]) for k in METRICS)<1e-10
     records.append(dict(panel=p,phase=phase,scenario=scenario,candidate=name,**m))
  for seed in [20261201,20261202,20261203,20261204]:
   fd=folds(A,seed)
   for k in range(3):
    for name,x in predict(A[fd!=k],c[fd!=k],base,p).items():cv.append(dict(panel=p,seed=seed,fold=k,candidate=name,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())))
  pd.DataFrame(records).to_csv(O/'trials.csv',index=False);pd.DataFrame(cv).to_csv(O/'real_count_cv.csv',index=False);print('completed',p,flush=True)
 tr=pd.DataFrame(records);counts=pd.DataFrame(cv);policy={};checks=[]
 for p in panels:
  t=tr[(tr.panel==p)&(tr.phase=='calibration')];cal=t.groupby('candidate')[METRICS].mean();ref=cal.loc['v11'];cc=counts[(counts.panel==p)&(counts.seed<=20261202)].groupby(['seed','candidate']).NMAE.mean().unstack();target=t[t.scenario.isin(TARGETS)].groupby(['scenario','candidate'])[METRICS].mean();eligible=['v11']
  for n in cal.index:
   delta=cal.loc[n]-ref;td=target.xs(n,level='candidate')-target.xs('v11',level='candidate');ratio=cc[n]/cc.v11;ok=bool((delta>=-1e-10).all() and (td>=-1e-10).all().all() and cc[n].mean()<=cc.v11.mean()*1.000001 and ratio.max()<=1.01)
   checks.append(dict(panel=p,candidate=n,eligible=ok,count_mean_ratio=float(cc[n].mean()/cc.v11.mean()),target_worst_Sdev_delta=float(td.S_dev.min()),**{'delta_'+k:float(delta[k]) for k in METRICS}))
   if ok:eligible.append(n)
  # Target priority fixed before evaluating heavy and sparse shapes.
  score=target.groupby('candidate').S_dev.mean();policy[p]=str(score.loc[list(dict.fromkeys(eligible))].idxmax())
 pd.DataFrame(checks).to_csv(O/'selection_checks.csv',index=False);(O/'policy.json').write_text(json.dumps({'policy':policy,'targets':TARGETS,'criterion':'target mean Sdev maximum; calibration all-four plus total nondecrease overall AND for each target family vs v11; grouped count CV mean no worse, worst selection seed ratio<=1.01','selection_seeds':[20261201,20261202],'benchmark_previously_seen':True},indent=2),encoding='utf-8')
 parts=[];cparts=[]
 for method,pol in [('v11',{p:'v11' for p in panels}),('v12',policy)]:
  for p,n in pol.items():
   q=tr[(tr.panel==p)&(tr.candidate==n)].copy();q['method']=method;parts.append(q)
   q=counts[(counts.panel==p)&(counts.candidate==n)].copy();q['method']=method;cparts.append(q)
 joined=pd.concat(parts);joined.to_csv(O/'comparison_trials.csv',index=False);joined.groupby(['phase','method'])[METRICS].mean().to_csv(O/'four_components_overall.csv');joined.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'four_components_by_scenario.csv');tr.groupby(['phase','candidate'])[METRICS].mean().to_csv(O/'all_candidates_overall.csv');pd.concat(cparts).groupby(['seed','method']).NMAE.mean().to_csv(O/'real_count_overall.csv');(O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'synthetic_cases':480,'candidates':tr.candidate.nunique(),'same_v10_benchmark':True,'v11_match_tolerance':1e-10,'CPU_threads':4,'hidden_truth_available':False,'baseline_preserved':True},indent=2),encoding='utf-8');print('POLICY',policy,flush=True);print(joined.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
if __name__=='__main__':main()
