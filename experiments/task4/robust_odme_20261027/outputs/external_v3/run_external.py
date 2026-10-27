# 中文说明：真实匝道统计只校准先验形状/方差；不建立未经证实的逐站对应。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent; B=O.parent;sys.path.insert(0,str(B));sys.path.insert(0,str(B/'improvement_v2'))
from run_task4 import REL,load,load_operator,solve as ridge
from run_v2 import folds,metric
import numpy as np,pandas as pd
from scipy.linalg import solve as linear_solve
NAMES=['official_005','external_variance','external_attr10','external_attr25','external_both10','external_both25','external_attr10_recent']
params=pd.read_csv(O/'external_parameters.csv').set_index('panel');real=pd.read_csv(O/'real_ramp_records.csv')
def prior(panel,b,paths,weight,recent=False):
 d=int(panel.split('_')[0][1:]);r=int(panel.split('_')[1][1:]);direction=panel[-1]
 subset=real[(real.district==d)&(real.route==r)&(real.direction==direction)&(real.ramp_type=='OFF')&(real.latest_volume>0)]
 if recent:
  cutoff=2022 if (subset.latest_year>=2022).sum()>=10 else subset.latest_year.max()-1
  subset=subset[subset.latest_year>=cutoff]
 vals=subset.latest_volume.to_numpy(float)
 agg=pd.Series(b).groupby(paths.destination_zone.to_numpy()).sum();order=agg.sort_values().index;v=np.quantile(vals,(np.arange(len(agg))+.5)/len(agg));v=v/v.sum()*agg.sum();target=pd.Series(v,index=order).reindex(agg.index)
 ratio=((1-weight)*agg+weight*target)/agg
 return b*paths.destination_zone.map(ratio).to_numpy()
def weighted(A,c,b,gamma):
 Q=A.toarray();d=.05*np.clip((np.median(b[b>0])/np.maximum(b,np.median(b[b>0])*.1))**gamma,1e-4,1e4);y=np.zeros(len(c))
 def fg(y):
  x=np.maximum(b-Q.T@y/d,0);return .5*np.dot(d*x,x)+c@y+.5*y@y,y+c-Q@x,x
 for i in range(200):
  val,g,x=fg(y)
  if abs(g).max()<1e-7:break
  H=(Q*(x>0)/d)@Q.T+np.eye(len(c));step=linear_solve(H,-g,assume_a='pos');rate=1.;slope=g@step
  for j in range(60):
   if fg(y+rate*step)[0]<=val+1e-4*rate*slope+1e-8:break
   rate*=.5
  y+=rate*step
 _,g,x=fg(y);assert abs(g).max()<1e-5,(i,abs(g).max());return x

def pred(panel,A,c,b,paths,name):
 if name=='official_005':return ridge(A,c,b,'ridge',.05)[0]
 alpha=.10 if '10' in name else .25
 bb=prior(panel,b,paths,alpha,recent=name.endswith('recent')) if 'attr' in name or 'both' in name else b
 gamma=float(params.loc[panel,'variance_exponent']) if 'variance' in name or 'both' in name else 0
 return weighted(A,c,bb,gamma)
def main():
 start=time.time();panels=sorted(p.name for p in (REL/'task4').iterdir());cv=[];cache={};fit=[];audit=[];syn=[]
 for panel in panels:
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,b,t,_=load(panel,'train',ids,links,D);cache[panel]=(ids,links,D,paths,A,c,b)
  fd=folds(A,20261029)
  for name in NAMES:
   for k in range(3):
    x=pred(panel,A[fd!=k],c[fd!=k],b,paths,name);cv.append({'panel':panel,'candidate':name,'fold':k,'NMAE':float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())})
  pd.DataFrame(cv).to_csv(O/'train_selection.csv',index=False);print('selection',panel,flush=True)
 scores=pd.DataFrame(cv).groupby(['candidate','panel']).NMAE.mean().groupby('candidate').mean().sort_values();chosen=str(scores.index[0]); ext=str(scores.drop('official_005').index[0]);names=list(dict.fromkeys(['official_005',chosen,ext]))
 (O/'selection.json').write_text(json.dumps({'chosen':chosen,'best_external':ext,'criterion':'train grouped heldout counts, not hidden path truth','scores':scores.to_dict()},indent=2),encoding='utf-8')
 frames={}
 for panel in panels:
  ids,links,D,paths,A,c,b=cache[panel];fd=folds(A,20261030)
  for name in names:
   for k in range(3):
    x=pred(panel,A[fd!=k],c[fd!=k],b,paths,name);audit.append({'panel':panel,'candidate':name,'fold':k,'NMAE':float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())})
  # 真实匝道跨年比率只用作合成扰动强度，不视为比赛隐藏标签。
  pairs=pd.read_csv(O/'real_ramp_year_pairs.csv');dist=int(panel.split('_')[0][1:]);route=int(panel.split('_')[1][1:]);p=pairs[(pairs.district==dist)&(pairs.route==route)&(pairs.direction==panel[-1])];ratios=np.clip((p.v1/p.v0).to_numpy(),.2,5)
  for seed in [8101,8102,8103]:
   rng=np.random.default_rng(seed);di=pd.factorize(paths.destination_zone)[0];f=b*rng.choice(ratios,di.max()+1)[di];cc=A@f
   for name in names:
    x=pred(panel,A,cc,b,paths,name);syn.append({'panel':panel,'seed':seed,'candidate':name,**metric(x,f,b,A,cc,paths.destination_zone.to_numpy())})
  for split in ['validation','private']:
   M,cc,bb,t,_=load(panel,split,ids,links,D)
   for name in names:
    x=pred(panel,M,cc,bb,paths,name);f=t.copy();f['path_flow']=x;frames.setdefault((split,name),[]).append(f)
    dest=paths.destination_zone.to_numpy();ah=pd.Series(x).groupby(dest).sum();ab=pd.Series(bb).groupby(dest).sum()
    fit.append({'panel':panel,'split':split,'candidate':name,'S_link':max(0,1-float(abs(M@x-cc).sum()/cc.sum())),'prior_relative_L1':float(abs(x-bb).sum()/bb.sum()),'destination_prior_TV':float(abs(ah/ah.sum()-ab/ab.sum()).sum()/2)})
  print('audit',panel,flush=True)
 for name in names:
  f=pd.concat([pd.concat(frames[(s,name)],ignore_index=True) for s in ['validation','private']],ignore_index=True);x=f.path_flow.to_numpy(float);assert np.isfinite(x).all() and (x>=0).all() and not f.duplicated(['panel','departure_time','path_id']).any();f.to_csv(O/f'task4_{name}.csv',index=False)
 pd.DataFrame(fit).to_csv(O/'fit_by_panel.csv',index=False);pd.DataFrame(fit).groupby(['split','candidate'])[['S_link','prior_relative_L1','destination_prior_TV']].mean().to_csv(O/'fit_overall.csv')
 pd.DataFrame(audit).to_csv(O/'resampling_audit.csv',index=False);pd.DataFrame(syn).to_csv(O/'real_calibrated_synthetic.csv',index=False);pd.DataFrame(syn).groupby('candidate')[['S_od','S_link','S_dev','S_attr','synthetic_S_ODME']].mean().to_csv(O/'real_calibrated_synthetic_overall.csv')
 (O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'synthetic_truth_is_competition_truth':False,'baseline_preserved':True},indent=2),encoding='utf-8');print('DONE',chosen,ext,flush=True)
if __name__=='__main__':main()
