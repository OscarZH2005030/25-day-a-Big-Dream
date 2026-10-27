"""Joint four-component guards; artificial truth and real count CV stay separate."""
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import sys, json, time
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=[str(O.parent/'robust_v9'),str(O.parent/'improvement_v2'),str(O.parent)]
from run_multi_error import scenario_input, SCENARIOS
from run_task4 import REL,load,load_operator,solve
from run_v2 import kl,metric,folds
import numpy as np
import pandas as pd
METRICS=['S_od','S_link','S_dev','S_attr','synthetic_S_ODME']
EXTRA=['correlated_path_prior','localized_undercount','rounded_counts','near_correct_prior']

def predict(A,c,b):
 x,st=solve(A,c,b,'ridge',.05);assert st['success']
 q,st=kl(A,c,b,.05);assert st['success']
 s,st=kl(A,c,b*c.sum()/max(float((A@b).sum()),1e-12),.05);assert st['success']
 out={'official':x}
 for label,y in [('raw',q),('scaled',s)]:
  for w in [.1,.25,.5,1.]:out[f'{label}_{w:g}']=(1-w)*x+w*y
 return out

def extra(f,A,paths,name,rng):
 floor=max(f.mean()*.02,1e-9);b=np.maximum(f,floor)*rng.lognormal(-.5*.35**2,.35,len(f));c=A@f
 if name=='correlated_path_prior':
  oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0]
  b=np.maximum(f,floor)*np.exp(.6*rng.normal(size=oi.max()+1)[oi]+.6*rng.normal(size=di.max()+1)[di])
 elif name=='localized_undercount':
  c=c.copy();start=int(rng.integers(len(c)));idx=(start+np.arange(max(1,len(c)//5)))%len(c);c[idx]*=.6
 elif name=='rounded_counts':c=np.maximum(np.round(c/50)*50,1.)
 elif name=='near_correct_prior':b=np.maximum(f,1e-9)*rng.lognormal(-.5*.03**2,.03,len(f))
 return f,b,c

def main():
 start=time.time();records=[];cv=[];cache={};panels=sorted(p.name for p in (REL/'task4').iterdir())
 for pn,p in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/p/'network');A,c,base,t,_=load(p,'train',ids,ls,D);cache[p]=(ids,ls,D,paths)
  for phase,offset,sigma in [('calibration',0,.3),('evaluation',10000,.8),('sparse_stress',20000,.8)]:
   rng=np.random.default_rng(101000+pn+offset);f0=base*c.sum()/(A@base).sum()*rng.lognormal(-.5*sigma*sigma,sigma,len(base))
   if phase=='sparse_stress':f0*=np.where(rng.random(len(base))<.15,6.4,.05)
   for sn,scenario in enumerate(SCENARIOS+EXTRA):
    rng=np.random.default_rng(102000+pn*100+sn+offset)
    f,b,cc=(scenario_input if scenario in SCENARIOS else extra)(f0.copy(),A,paths,scenario,rng)
    for name,x in predict(A,cc,b).items():records.append(dict(panel=p,phase=phase,scenario=scenario,candidate=name,**metric(x,f,b,A,cc,paths.destination_zone.to_numpy())))
  for seed in [20261201,20261202,20261203,20261204]:
   fd=folds(A,seed)
   for k in range(3):
    for name,x in predict(A[fd!=k],c[fd!=k],base).items():cv.append(dict(panel=p,seed=seed,fold=k,candidate=name,NMAE=float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())))
  pd.DataFrame(records).to_csv(O/'trials.csv',index=False);pd.DataFrame(cv).to_csv(O/'real_count_cv.csv',index=False);print('completed',p,flush=True)
 tr=pd.DataFrame(records);counts=pd.DataFrame(cv);policy={};checks=[]
 # Freeze using calibration and first two count seeds; no evaluation feedback.
 for p in panels:
  cal=tr[(tr.panel==p)&(tr.phase=='calibration')].groupby('candidate')[METRICS].mean();ref=cal.loc['official']
  cc=counts[(counts.panel==p)&(counts.seed<=20261202)].groupby(['seed','candidate']).NMAE.mean().unstack();eligible=['official']
  sc=tr[(tr.panel==p)&(tr.phase=='calibration')].groupby(['scenario','candidate']).synthetic_S_ODME.mean().unstack()
  for name in cal.index:
   if name=='official':continue
   delta=cal.loc[name]-ref;improve=1-cc[name].mean()/cc.official.mean();worst=(cc[name]/cc.official).max()
   ok=bool((delta[METRICS]>=-1e-10).all() and improve>=.005 and worst<=1.01 and (sc[name]-sc.official).min()>=-.005)
   checks.append(dict(panel=p,candidate=name,eligible=ok,real_count_improvement=improve,worst_seed_ratio=worst,**{'delta_'+k:float(delta[k]) for k in METRICS}))
   if ok:eligible.append(name)
  policy[p]=str(cal.loc[eligible].synthetic_S_ODME.idxmax())
 pd.DataFrame(checks).to_csv(O/'joint_selection_checks.csv',index=False)
 (O/'policy.json').write_text(json.dumps({'policy':policy,'selection':'all four calibration component means nondecreasing, total nondecreasing, real CV improvement>=0.5%, worst seed ratio<=1.01, worst scenario total delta>=-0.005','synthetic_only':True},indent=2),encoding='utf-8')
 selected=[]
 for p,n in policy.items():
  q=tr[(tr.panel==p)&(tr.candidate==n)].copy();q['method']='joint_policy';selected.append(q)
 q=tr[tr.candidate=='official'].copy();q['method']='official';selected.append(q)
 for p in panels:
  n='scaled_0.25' if p=='D7_I210_W' else 'official'
  q=tr[(tr.panel==p)&(tr.candidate==n)].copy();q['method']='previous_v9';selected.append(q)
 joined=pd.concat(selected);joined.groupby(['phase','method'])[METRICS].mean().to_csv(O/'four_components_overall.csv');joined.groupby(['phase','scenario','method'])[METRICS].mean().to_csv(O/'four_components_by_scenario.csv');joined.to_csv(O/'selected_trials.csv',index=False)
 selcv=[]
 for p,n in policy.items():
  for method,name in [('official','official'),('joint_policy',n)]:
   q=counts[(counts.panel==p)&(counts.candidate==name)].copy();q['method']=method;selcv.append(q)
 pd.concat(selcv).groupby(['seed','method']).NMAE.mean().to_csv(O/'selected_real_count_cv.csv')
 # All components for actual published counts: explicitly unknown path components.
 actual=[];frames=[]
 for p,(ids,ls,D,paths) in cache.items():
  for split in ['validation','private']:
   A,c,b,t,_=load(p,split,ids,ls,D);sols=predict(A,c,b)
   for method,n in [('official','official'),('joint_policy',policy[p])]:actual.append(dict(panel=p,split=split,method=method,S_od=None,S_link=max(0,1-float(abs(A@sols[n]-c).sum()/c.sum())),S_dev=None,S_attr=None,S_ODME=None,truth_available=False))
   t=t.copy();t['path_flow']=sols[policy[p]];frames.append(t)
 final=pd.concat(frames,ignore_index=True);base=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str);keys=['panel','departure_time','path_id','origin_zone','destination_zone'];assert final[keys].astype(str).sort_values(keys).reset_index(drop=True).equals(base[keys].sort_values(keys).reset_index(drop=True));assert np.isfinite(final.path_flow).all() and (final.path_flow>=0).all();final.to_csv(O/'task4_joint_candidate.csv',index=False);pd.DataFrame(actual).to_csv(O/'actual_four_components.csv',index=False)
 tr.groupby(['phase','candidate'])[METRICS].mean().to_csv(O/'all_candidates_four_components.csv')
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'synthetic_cases':len(tr)//9,'candidates':9,'components':METRICS,'real_count_selection_seeds':[20261201,20261202],'real_count_audit_seeds':[20261203,20261204],'hidden_truth_available':False,'baseline_preserved':True,'policy':policy},indent=2),encoding='utf-8');print(joined.groupby(['phase','method'])[METRICS].mean().to_string(),flush=True)
if __name__=='__main__':main()
