# 中文说明：多类弱先验/计数误差、分阶段复核和不识别反例；所有人工真值仅用于敏感性实验。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent));sys.path.insert(0,str(O.parent/'improvement_v2'))
from run_task4 import REL,load,load_operator,solve as ridge
from run_v2 import kl,metric,folds
import numpy as np,pandas as pd
from scipy.linalg import solve as linear_solve
NAMES=['official','amp125','scaled_kl','mix25','mix50','mix75','noise03','noise10']
SCENARIOS=['unbiased_prior','prior_over2','prior_under05','origin_bias','destination_bias','additive_prior','permuted_prior','count_noise05','count_outliers','count_systematic30','spatial_count_tilt','invisible_redistribution']

def noise_solver(A,c,b,cv):
 scale=float(np.median(c/np.maximum(A@b,1e-9)));p=np.maximum(b*scale,1e-9);Q=A.toarray()/np.maximum(cv*c,cv*np.median(c)*.1)[:,None];cc=c/np.maximum(cv*c,cv*np.median(c)*.1);d=1/np.maximum(.75*p,.075*np.median(p))**2;y=np.zeros(len(c))
 def fg(y):
  x=np.maximum(p-Q.T@y/d,0);return .5*np.dot(d*x,x)+cc@y+.5*y@y,y+cc-Q@x,x
 for i in range(200):
  val,g,x=fg(y)
  if abs(g).max()<1e-7:break
  H=(Q*(x>0)/d)@Q.T+np.eye(len(c));step=linear_solve(H,-g,assume_a='pos');a=1.;slope=g@step
  for _ in range(60):
   if fg(y+a*step)[0]<=val+1e-4*a*slope+1e-9:break
   a*=.5
  y+=a*step
 _,g,x=fg(y);assert abs(g).max()<1e-5,(i,abs(g).max());return x

def predict(A,c,b):
 off,st=ridge(A,c,b,'ridge',.05);assert st['success'];y,st=kl(A,c,b,.05);assert st['success'];amp,st=ridge(A,c,np.maximum(b+1.25*(y-b),0),'ridge',.05);assert st['success'];s=c.sum()/(A@b).sum();q,st=kl(A,c,b*s,.05);assert st['success']
 return {'official':off,'amp125':amp,'scaled_kl':q,'mix25':.75*off+.25*q,'mix50':.5*(off+q),'mix75':.25*off+.75*q,'noise03':noise_solver(A,c,b,.03),'noise10':noise_solver(A,c,b,.1)}

def scenario_input(f0,A,paths,scenario,rng):
 f=f0.copy();oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0];floor=max(np.mean(f)*.02,1e-8);b=np.maximum(f,floor)*rng.lognormal(-.5*.35**2,.35,len(f));c=A@f
 if scenario=='prior_over2':b*=2
 elif scenario=='prior_under05':b*=.5
 elif scenario=='origin_bias':b=np.maximum(f,floor)*np.exp(rng.normal(0,.65,oi.max()+1)[oi])
 elif scenario=='destination_bias':b=np.maximum(f,floor)*np.exp(rng.normal(0,.65,di.max()+1)[di])
 elif scenario=='additive_prior':b=np.maximum(f+rng.normal(0,.5*np.mean(f),len(f)),floor)
 elif scenario=='permuted_prior':b=np.maximum(rng.permutation(f),floor)
 elif scenario=='count_noise05':c=c*rng.lognormal(-.5*.05**2,.05,len(c))
 elif scenario=='count_outliers':c=c.copy();idx=rng.choice(len(c),max(1,len(c)//20),replace=False);c[idx]*=2.
 elif scenario=='count_systematic30':b=np.maximum(f,floor)*rng.lognormal(-.5*.1**2,.1,len(f));c*=1.3
 elif scenario=='spatial_count_tilt':b=np.maximum(f,floor)*rng.lognormal(-.5*.1**2,.1,len(f));c*=np.linspace(.7,1.3,len(c))
 elif scenario=='invisible_redistribution':
  # OD四边形环流保持每个link的加载完全相同；该情景揭示计数无法识别的真实不确定性。
  b=f.copy();lut={(int(o[1:]),int(d[1:])):i for i,(o,d) in enumerate(zip(paths.origin_zone,paths.destination_zone))};zones=sorted(set(int(z[1:]) for z in paths.origin_zone)|set(int(z[1:]) for z in paths.destination_zone))
  for _ in range(400):
   i,j,k,l=sorted(rng.choice(zones,4,replace=False));pairs=[(i,k),(j,l),(i,l),(j,k)]
   if all(p in lut for p in pairs):
    a,bb,dd,e=[lut[p] for p in pairs];delta=.7*min(f[a],f[bb]);f[a]-=delta;f[bb]-=delta;f[dd]+=delta;f[e]+=delta
  assert np.max(abs(A@f-c))<1e-7
 return f,np.maximum(b,1e-9),c

def main():
 start=time.time();records=[];cache={};panels=sorted(p.name for p in (REL/'task4').iterdir())
 for pnum,panel in enumerate(panels):
  ids,ls,D,paths=load_operator(REL/'corridors'/panel/'network');A,c,base,t,_=load(panel,'train',ids,ls,D);cache[panel]=(ids,ls,D,paths)
  for phase,sigma in [('calibration',.25),('evaluation',.65),('sparse_stress',.65)]:
   rng0=np.random.default_rng(93000+pnum+(0 if phase=='calibration' else 100 if phase=='evaluation' else 200));f0=base*(c.sum()/(A@base).sum())*rng0.lognormal(-.5*sigma*sigma,sigma,len(base))
   if phase=='sparse_stress':f0*=np.where(rng0.random(len(f0))<.2,4.8,.05)
   for sn,scenario in enumerate(SCENARIOS):
    rng=np.random.default_rng(94000+1000*pnum+sn+(0 if phase=='calibration' else 100 if phase=='evaluation' else 200));f,b,cc=scenario_input(f0,A,paths,scenario,rng);sols=predict(A,cc,b)
    for name,x in sols.items():records.append({'panel':panel,'phase':phase,'scenario':scenario,'candidate':name,**metric(x,f,b,A,cc,paths.destination_zone.to_numpy())})
  pd.DataFrame(records).to_csv(O/'multi_error_trials.csv',index=False);print('multi-error',panel,flush=True)
 df=pd.DataFrame(records);cal=df[df.phase=='calibration'].groupby(['scenario','candidate'])[['S_od','S_dev','S_attr','synthetic_S_ODME']].mean();score=cal.synthetic_S_ODME.unstack();regret=score.max(axis=1).to_numpy()[:,None]-score.to_numpy();reg=pd.Series(regret.max(axis=0),index=score.columns);macro=cal.groupby('candidate').mean();ref=macro.loc['official'];eligible=macro[(macro.S_dev>=ref.S_dev)&(macro.S_od>=ref.S_od-.005)&(macro.S_attr>=ref.S_attr-.002)]
 chosen=str(reg.reindex(eligible.index).idxmin()) if len(eligible) else 'official'
 # 留出整个误差族：每次选模排除该族，只在剩余calibration族做同样约束。
 held=[]
 for scenario in SCENARIOS:
  others=cal.loc[cal.index.get_level_values('scenario')!=scenario];mm=others.groupby('candidate').mean();r=mm.loc['official'];ee=mm[(mm.S_dev>=r.S_dev)&(mm.S_od>=r.S_od-.005)&(mm.S_attr>=r.S_attr-.002)];ss=others.synthetic_S_ODME.unstack();rr=(ss.max(axis=1).to_numpy()[:,None]-ss.to_numpy()).max(axis=0);rr=pd.Series(rr,index=ss.columns);pick=str(rr.reindex(ee.index).idxmin()) if len(ee) else 'official'
  for phase in ['evaluation','sparse_stress']:
   g=df[(df.phase==phase)&(df.scenario==scenario)];
   for method,cand in [('leave_error_family_out',pick),('official','official'),('previous_scaled_kl','scaled_kl')]:
    for _,row in g[g.candidate==cand].iterrows():held.append({'panel':row.panel,'phase':phase,'scenario':scenario,'method':method,'chosen':cand,**{k:row[k] for k in ['S_od','S_dev','S_attr','synthetic_S_ODME']}})
 df.groupby(['phase','candidate'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'multi_error_overall.csv');df.groupby(['phase','scenario','candidate'])[['S_od','S_dev','S_attr','S_link','synthetic_S_ODME']].mean().to_csv(O/'multi_error_by_scenario.csv');pd.DataFrame(held).to_csv(O/'leave_error_family_out.csv',index=False)
 (O/'robust_selection.json').write_text(json.dumps({'chosen':chosen,'criterion':'calibration minimum worst error-family SODME regret, with Sdev/path/attraction guards vs official','calibration_macro':macro.to_dict(orient='index'),'max_family_regret':reg.to_dict(),'synthetic_only':True,'phases':'calibration mild shapes, evaluation heavy shapes, sparse_stress; calibration choices fixed before evaluation aggregation'},indent=2),encoding='utf-8')
 # 真实计数三种独立分组随机划分，全部十方向；不把代理验证分数当作真实路径评分。
 counts=[];frames=[];baseframe=pd.read_csv(O.parent/'task4_path_flow.csv',dtype=str)
 for panel,(ids,ls,D,paths) in cache.items():
  A,c,b,t,_=load(panel,'train',ids,ls,D)
  for seed in [20261103,20261104,20261105]:
   fd=folds(A,seed)
   for k in range(3):
    sols=predict(A[fd!=k],c[fd!=k],b)
    for name in NAMES:counts.append({'panel':panel,'seed':seed,'fold':k,'candidate':name,'NMAE':float(abs(A[fd==k]@sols[name]-c[fd==k]).sum()/c[fd==k].sum())})
  for split in ['validation','private']:
   A,c,b,t,_=load(panel,split,ids,ls,D);sols=predict(A,c,b);t=t.copy();t['path_flow']=sols[chosen];frames.append(t)
  pd.DataFrame(counts).to_csv(O/'actual_train_count_cv.csv',index=False);print('real-count validation',panel,flush=True)
 final=pd.concat(frames,ignore_index=True);keys=['panel','departure_time','path_id','origin_zone','destination_zone'];a=final[keys].sort_values(keys).reset_index(drop=True);bb=baseframe[keys].sort_values(keys).reset_index(drop=True);v=final.path_flow.to_numpy(float);assert a.equals(bb) and np.isfinite(v).all() and (v>=0).all() and not final.duplicated(keys[:3]).any();final.to_csv(O/'task4_robust_candidate.csv',index=False)
 pd.DataFrame(counts).groupby(['seed','candidate']).NMAE.mean().to_csv(O/'actual_count_cv_overall.csv');(O/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'candidate':chosen,'panels':10,'error_families':12,'synthetic_cases':360,'rows':len(final),'all_keys_match':True,'finite_nonnegative':True,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'baseline_preserved':True,'hidden_path_truth_used':False},indent=2),encoding='utf-8');print('DONE',chosen,time.time()-start)
if __name__=='__main__':main()
