# 中文说明：仅使用公开计数和弱先验；不构造或读取隐藏路径标签。
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import sys,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from scipy.optimize import minimize, lsq_linear
sys.path.insert(0,str(Path(__file__).resolve().parent/'official_reference/src'))
from task4.build_task4_odme_artifacts import load_operator
ROOT=Path(__file__).resolve().parents[1]
REL=Path(os.environ['TFB_RELEASE_ROOT']).expanduser().resolve()
OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
CAND={'official_005':('ridge',.05),'ridge_05':('ridge',.5),'relative_005':('relative',.05),'relative_05':('relative',.5)}
def solve(A,c,b,kind,lam):
 # 对偶半光滑牛顿法：仅解观测数阶系统，精确保留非负约束及先验正则。
 from scipy.linalg import solve as linear_solve
 w=np.ones(len(b)) if kind=='ridge' else np.median(b[b>0])/np.maximum(b,np.median(b[b>0])*.1)
 d=lam*w; Q=A.toarray(); y=np.zeros(len(c))
 def fg(y):
  x=np.maximum(b-Q.T@y/d,0)
  return .5*np.dot(d*x,x)+c@y+.5*y@y,y+c-Q@x,x
 for i in range(200):
  val,g,x=fg(y)
  if abs(g).max()<1e-7: break
  H=(Q*(x>0)/d)@Q.T+np.eye(len(c))
  step=linear_solve(H,-g,assume_a='pos'); slope=g@step; alpha=1.
  for j in range(60):
   if fg(y+alpha*step)[0]<=val+1e-4*alpha*slope+1e-8: break
   alpha*=.5
  y+=alpha*step
 val,g,x=fg(y); grad=np.asarray(A.T@(A@x-c)).ravel()+d*(x-b)
 pg=np.where((x<=1e-8)&(grad>0),0,grad)
 return x,{'success':bool(abs(g).max()<1e-5),'iterations':i+1,'projected_gradient_inf':float(abs(pg).max()),'message':'dual semismooth Newton; dual_gradient_inf='+str(float(abs(g).max()))}

def load(panel,split,ids,links,A):
 d=REL/'task4'/panel/split
 cf=pd.read_csv(d/'synthetic_link_counts.csv',dtype={'link_id':str}); bf=pd.read_csv(d/'synthetic_weak_prior.csv',dtype=str); t=pd.read_csv(d/'sample_submission_path_flow.csv',dtype=str)
 assert not cf.link_id.duplicated().any() and not bf.path_id.duplicated().any()
 ix=[i for i,l in enumerate(links) if l in set(cf.link_id)]
 c=cf.set_index('link_id').reindex([links[i] for i in ix])['count'].to_numpy(float)
 b=bf.set_index('path_id').reindex(ids).path_flow.to_numpy(float)
 assert np.isfinite(c).all() and np.isfinite(b).all() and (c>=0).all() and (b>=0).all()
 assert t.path_id.tolist()==ids
 return sparse.csr_matrix(A[ix]),c,b,t,cf

def main():
 start=time.time(); audits=[]; cv=[]; diagnostics=[]; frames={}; statuses=[]
 checkpoint=OUT/'train_count_holdout.csv'
 fingerprint=hashlib.sha256(Path(__file__).read_bytes())
 for source in sorted((REL/'task4').rglob('*.csv'))+sorted((REL/'corridors').glob('*/network/path*.csv')):
  fingerprint.update(source.read_bytes())
 signature=fingerprint.hexdigest(); meta=OUT/'checkpoint_signature.txt'
 old=pd.read_csv(checkpoint) if checkpoint.exists() and meta.exists() and meta.read_text()==signature else pd.DataFrame()
 panels=sorted(p.name for p in (REL/'task4').iterdir())
 for panel in panels:
  ids,links,dense,paths=load_operator(REL/'corridors'/panel/'network'); A,c,b,t,cf=load(panel,'train',ids,links,dense)
  # 相同观测方程必须留在同一折，防止重复探测器泄漏。
  signatures=[hashlib.sha256(A.getrow(i).toarray().tobytes()).hexdigest() for i in range(A.shape[0])]
  unique=sorted(set(signatures)); rng=np.random.default_rng(20261026); rng.shuffle(unique); groups={s:i%3 for i,s in enumerate(unique)}; fold=np.array([groups[s] for s in signatures])
  rank=int(np.linalg.matrix_rank(A.toarray()))
  audits.append({'panel':panel,'paths':len(ids),'operator_links':len(links),'train_counted_links':len(c),'unique_equations':len(unique),'rank':rank,'nullity':len(ids)-rank,'unobserved_paths':int((np.asarray(A.sum(axis=0)).ravel()==0).sum()),'counts_outside_operator':len(set(cf.link_id)-set(links)),'template_all_zero':bool((t.path_flow.astype(float)==0).all()),'period':t.departure_time.unique().tolist()})
  for name,(kind,lam) in CAND.items():
   for k in range(3):
    prior_row=old[(old.panel==panel)&(old.candidate==name)&(old.fold==k)&(old.success==True)] if len(old) else old
    if len(prior_row):
     cv.append(prior_row.iloc[0].to_dict()); continue
    x,st=solve(A[fold!=k],c[fold!=k],b,kind,lam)
    cv.append({'panel':panel,'candidate':name,'fold':k,'heldout_n':int((fold==k).sum()),'heldout_nmae':float(abs(A[fold==k]@x-c[fold==k]).sum()/c[fold==k].sum()),**st})
  pd.DataFrame(cv).to_csv(OUT/'train_count_holdout.csv',index=False)
  meta.write_text(signature)
  print('CV finished',panel,flush=True)
 scores=pd.DataFrame(cv).groupby(['candidate','panel']).heldout_nmae.mean().groupby('candidate').mean().sort_values()
 assert pd.DataFrame(cv).success.all(), '存在未收敛候选，禁止选参'; chosen=str(scores.index[0]); (OUT/'selection.json').write_text(json.dumps({'criterion':'train only, grouped 3-fold heldout count NMAE; not path-flow truth','scores':scores.to_dict(),'chosen':chosen},indent=2),encoding='utf-8')
 for panel in panels:
  ids,links,dense,paths=load_operator(REL/'corridors'/panel/'network')
  for split in ['train','validation','private']:
   A,c,b,t,cf=load(panel,split,ids,links,dense)
   for name in CAND:
    x,st=solve(A,c,b,*CAND[name]); assert st['success'],st; f=t.copy(); f['path_flow']=x
    assert np.isfinite(x).all() and (x>=0).all() and not f.duplicated(['panel','departure_time','path_id']).any()
    frames.setdefault((name,split),[]).append(f)
    dest=paths.destination_zone.to_numpy(); ah=pd.Series(x).groupby(dest).sum(); ab=pd.Series(b).groupby(dest).sum()
    diagnostics.append({'panel':panel,'split':split,'candidate':name,'S_link':max(0,1-float(abs(A@x-c).sum()/c.sum())),'prior_L1':float(abs(x-b).sum()),'prior_relative_L1':float(abs(x-b).sum()/b.sum()),'destination_prior_TV':float(abs(ah/ah.sum()-ab/ab.sum()).sum()/2),'total_flow':float(x.sum()),**st})
  print('predictions finished',panel,flush=True)
 for (name,split),fs in frames.items(): pd.concat(fs,ignore_index=True).to_csv(OUT/f'{name}_{split}.csv',index=False)
 final=pd.concat([pd.concat(frames[(chosen,s)],ignore_index=True) for s in ['validation','private']],ignore_index=True); final.to_csv(OUT/'task4_path_flow.csv',index=False)
 pd.DataFrame(audits).to_json(OUT/'audit.json',orient='records',indent=2,force_ascii=False)
 pd.DataFrame(diagnostics).to_csv(OUT/'comparison_by_panel.csv',index=False)
 pd.DataFrame(diagnostics).groupby(['split','candidate'])[['S_link','prior_relative_L1','destination_prior_TV']].mean().to_csv(OUT/'comparison_overall.csv')
 (OUT/'run_manifest.json').write_text(json.dumps({'official_commit':'f9e3fbc35c066419726987e470448f33d4c72fd9','elapsed_seconds':time.time()-start,'CPU_threads':4,'CPU_temperature':None,'GPU_used':False,'selected':chosen,'final_rows':len(final),'path_truth_available':False},indent=2),encoding='utf-8')
 print('DONE',chosen,len(final),time.time()-start,flush=True)
if __name__=='__main__': main()
