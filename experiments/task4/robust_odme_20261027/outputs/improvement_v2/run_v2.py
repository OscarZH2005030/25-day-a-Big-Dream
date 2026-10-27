# 中文说明：第二轮候选仅用公开train计数选参；合成真值只作假设敏感性实验。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from run_task4 import REL,load,load_operator,solve as ridge
import numpy as np,pandas as pd
from scipy.linalg import solve as linear_solve
OUT=Path(__file__).resolve().parent
NAMES=['official_005','kl_0005','kl_005','kl_05','scaled_ridge','blend_kl_50']
def kl(A,c,b,alpha):
 Q=A.toarray(); tau=alpha*np.median(b[b>0]); y=np.zeros(len(c))
 def fg(y):
  x=np.maximum(b,1e-12)*np.exp(np.clip(-Q.T@y/tau,-600,600))
  return tau*x.sum()+c@y+.5*y@y,y+c-Q@x,x
 for it in range(150):
  val,g,x=fg(y)
  if abs(g).max()<1e-7:break
  H=(Q*(x/tau))@Q.T+np.eye(len(c)); step=linear_solve(H,-g,assume_a='pos'); slope=g@step; rate=1.
  for j in range(60):
   if fg(y+rate*step)[0]<=val+1e-4*rate*slope+1e-7:break
   rate*=.5
  y+=rate*step
 val,g,x=fg(y)
 return x,{'success':bool(abs(g).max()<2e-5),'iterations':it+1,'residual':float(abs(g).max())}
def predict(A,c,b,name):
 if name=='official_005':return ridge(A,c,b,'ridge',.05)
 if name=='scaled_ridge':
  scale=c.sum()/max(float((A@b).sum()),1e-12)
  return ridge(A,c,b*scale,'ridge',.05)
 if name=='blend_kl_50':
  x,st=ridge(A,c,b,'ridge',.05); y,s=kl(A,c,b,.05)
  return .5*(x+y),{'success':st['success'] and s['success']}
 return kl(A,c,b,{'kl_0005':.005,'kl_005':.05,'kl_05':.5}[name])
def folds(A,seed):
 sig=[hashlib.sha256(A.getrow(i).toarray().tobytes()).hexdigest() for i in range(A.shape[0])]; u=sorted(set(sig));np.random.default_rng(seed).shuffle(u); mp={s:i%3 for i,s in enumerate(u)}
 return np.array([mp[s] for s in sig])
def metric(x,f,b,A,c,dest):
 dh=abs(x-b).sum();dt=abs(f-b).sum()
 ah=pd.Series(x).groupby(dest).sum();at=pd.Series(f).groupby(dest).sum()
 od=max(0,1-abs(x-f).sum()/f.sum());li=max(0,1-abs(A@x-c).sum()/c.sum());dev=np.exp(-abs(dh/max(dt,1e-12)-1));attr=max(0,1-abs(ah/ah.sum()-at/at.sum()).sum()/2)
 return {'S_od':od,'S_link':li,'S_dev':dev,'S_attr':attr,'synthetic_S_ODME':.45*od+.25*li+.15*dev+.15*attr}
def main():
 start=time.time();panels=sorted(p.name for p in (REL/'task4').iterdir());cv=[];audit=[];synthetic=[];fit=[];cache={}
 for panel in panels:
  ids,links,D,paths=load_operator(REL/'corridors'/panel/'network'); A,c,b,t,cf=load(panel,'train',ids,links,D); cache[panel]=(ids,links,D,paths,A,c,b)
  fd=folds(A,20261027)
  for name in NAMES:
   for k in range(3):
    x,st=predict(A[fd!=k],c[fd!=k],b,name);assert st['success'],(panel,name,st)
    cv.append({'panel':panel,'candidate':name,'fold':k,'NMAE':float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())})
  pd.DataFrame(cv).to_csv(OUT/'selection_count_cv.csv',index=False);print('selection',panel,flush=True)
 scores=pd.DataFrame(cv).groupby(['candidate','panel']).NMAE.mean().groupby('candidate').mean().sort_values();best=str(scores.index[0]); short=list(dict.fromkeys(['official_005',best,'kl_005','scaled_ridge']))
 (OUT/'selection.json').write_text(json.dumps({'seed':20261027,'criterion':'train grouped count holdout NMAE; not path truth','chosen':best,'scores':scores.to_dict(),'synthetic_does_not_select':True},indent=2),encoding='utf-8')
 for panel in panels:
  ids,links,D,paths,A,c,b=cache[panel];dest=paths.destination_zone.to_numpy();fd=folds(A,20261028)
  for name in list(dict.fromkeys(['official_005',best])):
   for k in range(3):
    x,st=predict(A[fd!=k],c[fd!=k],b,name);assert st['success']
    audit.append({'panel':panel,'candidate':name,'fold':k,'NMAE':float(abs(A[fd==k]@x-c[fd==k]).sum()/c[fd==k].sum())})
  # 人工构造三种不同的先验误差，检验方法是否只适合某一种假设。
  for scenario in ['path_multiplicative','origin_destination','additive_clipped']:
   rng=np.random.default_rng(7301+panels.index(panel))
   if scenario=='path_multiplicative':f=b*np.exp(rng.normal(-.125,.5,len(b)))
   elif scenario=='origin_destination':
    oi=pd.factorize(paths.origin_zone)[0];di=pd.factorize(paths.destination_zone)[0];f=b*np.exp(rng.normal(-.0625,.25,oi.max()+1)[oi]+rng.normal(-.0625,.25,di.max()+1)[di])
   else:f=np.maximum(0,b+rng.normal(0,.35*np.mean(b),len(b)))
   counts=A@f
   for name in short:
    x,st=predict(A,counts,b,name);assert st['success'];synthetic.append({'panel':panel,'scenario':scenario,'candidate':name,**metric(x,f,b,A,counts,dest)})
  for split in ['validation','private']:
   M,cc,bb,tt,_=load(panel,split,ids,links,D)
   for name in short:
    x,st=predict(M,cc,bb,name);assert st['success'];ff=tt.copy();ff['path_flow']=x
    assert np.isfinite(x).all() and (x>=0).all(); ff.to_csv(OUT/f'{panel}_{split}_{name}.csv',index=False)
    fit.append({'panel':panel,'split':split,'candidate':name,'S_link':max(0,1-float(abs(M@x-cc).sum()/cc.sum())),'prior_relative_L1':float(abs(x-bb).sum()/bb.sum()),'success':st['success']})
  pd.DataFrame(synthetic).to_csv(OUT/'synthetic_assumption_tests.csv',index=False);print('audit and sensitivity',panel,flush=True)
 pd.DataFrame(audit).to_csv(OUT/'resampling_audit.csv',index=False); pd.DataFrame(fit).to_csv(OUT/'fit_by_panel.csv',index=False)
 pd.DataFrame(synthetic).groupby(['scenario','candidate'])[['S_od','S_link','S_dev','S_attr','synthetic_S_ODME']].mean().to_csv(OUT/'synthetic_assumption_overall.csv')
 pd.DataFrame(fit).groupby(['split','candidate'])[['S_link','prior_relative_L1']].mean().to_csv(OUT/'fit_overall.csv')
 for name in short:
  fs=[pd.read_csv(OUT/f'{p}_{s}_{name}.csv',dtype={'path_id':str}) for s in ['validation','private'] for p in panels];pd.concat(fs,ignore_index=True).to_csv(OUT/f'task4_{name}.csv',index=False)
 (OUT/'manifest.json').write_text(json.dumps({'seconds':time.time()-start,'chosen_on_proxy':best,'CPU_threads':4,'GPU_used':False,'CPU_temperature':None,'truth_available':False,'baseline_frozen':True},indent=2),encoding='utf-8');print('DONE',best,flush=True)
if __name__=='__main__':main()
