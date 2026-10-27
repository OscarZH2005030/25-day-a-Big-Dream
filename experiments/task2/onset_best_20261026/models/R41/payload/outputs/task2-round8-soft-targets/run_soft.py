"""含噪阈值标签的软概率目标消融；完全复用第五/六轮监督输入及采样权重。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';sys.path[1:1]=[str(R7),str(R6),str(R5),str(R2)]
from structured_runtime import StructuredPredictor
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from special_features import features
from ramp_features import extras
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEED=20261027;TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def guard(stage):
 m=json.loads((TASK1/'package-manifest.json').read_text());h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=92,hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def matrix(s):
 r,p=extras(s);return np.concatenate([features(s),r,p],axis=1)
def fit(name,X,y,w,leaves):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%200==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=600)
 model=lgb.train(dict(objective='cross_entropy',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=600-done,init_model=initial,callbacks=[callback]) if done<600 else initial;model.save_model(str(path));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps) if s['has_future_queue']]))
def main():
 started=time.time();assert (R7/'final-verification.json').exists();before=guard('before');baseline=StructuredPredictor(R7);bt=baseline.rule['thresholds'];entries=json.loads((R2/'expanded-fit.json').read_text())['samples'];tune=role_samples('tune');models={};routes={};development=[];audit=[]
 save('experiment.json',dict(seed=SEED,models=32,rounds_per_model=600,max_tree_rounds=19200,cpu_threads=4,GPU_training=False,objective='cross_entropy: supports probability labels [0,1], not binary thresholding of soft labels',target='sigmoid((v_cut - eligible noisy future speed_kmh) / sigma_kmh)',sigmas_kmh=[2.5,5.],inputs='exact same 555D mainline/ramp/partial-physics',sampling='exact same row IDs and IPW weights from round5/6',selection='December per-road-condition with half-month stability; January global blend/threshold; February repeated report only',Task1='92-file frozen guard, historical .942828633 not remeasured',official_S_queue=None,documentation='https://github.com/lightgbm-org/LightGBM/blob/main/docs/Parameters.rst'))
 causal=[]
 for p in PANELS:
  for c in [0,1]:
   e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
   for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
   v['eligible'][slot+1:]=False;s2=make_sample(v,s['network'],slot,p);assert np.array_equal(matrix(s),matrix(s2));causal.append(dict(panel=p,condition=c,passed=True))
 save('causality-check.json',dict(passed=True,groups=causal,soft_label_margin_is_target_only=True,ramps_retain_Tminus5_cutoff=True))
 for p in PANELS:
  for c in [0,1]:
   tag=f'{p}-{c}';progress('building_soft_targets',panel=p,condition=c,completed=len(routes),total=16);basepath=ROOT.parents[1]/'work/round5-distribution'/tag;ramppath=ROOT.parents[1]/'work/round6-ramp-matrix'/tag;x=np.load(basepath/'X.npy',mmap_mode='r');hard=np.load(basepath/'y.npy',mmap_mode='r');w=np.load(basepath/'w.npy',mmap_mode='r');residual=np.load(basepath/'residual.npy',mmap_mode='r');rr=np.load(ramppath/'ramp.npy',mmap_mode='r');ph=np.load(ramppath/'physics.npy',mmap_mode='r');future_ratio=np.asarray(residual)+x[:,175];free=x[:,171]*130;cut_ratio=x[:,173];margin=(cut_ratio-future_ratio)*free;away=np.abs(margin)>.001;assert np.array_equal((margin[away]>=0).astype('float32'),hard[away]);xx=np.concatenate([x,rr,ph],axis=1);item=dict(panel=p,condition=c,rows=len(hard),features=xx.shape[1],threshold_sign_matches_hard_labels_away_from_float_boundary=True,near_boundary_cells=int((np.abs(margin)<5).sum()),near_boundary_fraction=float((np.abs(margin)<5).mean()));audit.append(item);save('soft-label-audit.json',audit)
   for sigma in [2.5,5.]:soft=(1/(1+np.exp(np.clip(-margin/sigma,-30,30)))).astype('float32');models[(p,c,sigma)]=fit(f'{tag}-sigma{sigma}',xx,soft,w,31 if c==0 else 63)
   del xx,x,rr,ph;gc.collect();ss=[s for s in tune if s['panel']==p and s['condition']==c];matrices=[matrix(s) for s in ss];b=[baseline.probability(s) for s in ss];bv=metric(ss,b,bt[c]);half=[[i for i,s in enumerate(ss) if (pd.Timestamp(s['origin']).day<=15)==first] for first in [True,False]];candidates=[dict(sigma=None,iteration=None,alpha_new=0.,proxy=bv,half_gains=[0.,0.],eligible=True)]
   for sigma in [2.5,5.]:
    for iteration in [200,400,600]:
     raw=[models[(p,c,sigma)].predict(x,num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1) for x in matrices]
     for alpha in [.25,.5,1.]:
      ps=[alpha*a+(1-alpha)*bb for a,bb in zip(raw,b)];value=metric(ss,ps,bt[c]);gains=[metric([ss[i] for i in ii],[ps[i] for i in ii],bt[c])-metric([ss[i] for i in ii],[b[i] for i in ii],bt[c]) for ii in half];candidates.append(dict(sigma=sigma,iteration=iteration,alpha_new=alpha,proxy=value,half_gains=gains,eligible=value>=bv+.003 and min(gains)>=-.02))
   chosen=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[(p,c)]=chosen;development.append(dict(panel=p,condition=c,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',development);save('routes.json',[dict(panel=p,condition=c,**r) for (p,c),r in routes.items()]);del matrices;gc.collect()
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);r=routes[(s['panel'],s['condition'])];base.append(b)
   if r['sigma'] is None:adapt.append(b)
   else:
    raw=models[(s['panel'],s['condition'],r['sigma'])].predict(matrix(s),num_iteration=r['iteration'],num_threads=4).astype('float32').reshape(6,-1);adapt.append(r['alpha_new']*raw+(1-r['alpha_new'])*b)
  return base,adapt
 progress('calibration');cal=role_samples('calibrate');base,adapt=components(cal);candidates=[dict(alpha_soft=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.25,.5,.75,1.]:
  ps=[alpha*a+(1-alpha)*b for a,b in zip(adapt,base)]
  for t0 in [.075,.1,.15,.2,.25,.35,.5]:
   for t1 in [.075,.1,.15,.2,.25,.35,.5]:candidates.append(dict(alpha_soft=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((R7/'frozen-calibration.json').read_bytes()).hexdigest(),february='reused report only',Task1_frozen=True))
 progress('evaluating');audit_samples=role_samples('audit');base,adapt=components(audit_samples);final=[chosen['alpha_soft']*a+(1-chosen['alpha_soft'])*b for a,b in zip(adapt,base)];reports={'round7_frozen':score(audit_samples,base,bt),'soft_same_thresholds':score(audit_samples,adapt,bt),'selected':score(audit_samples,final,chosen['thresholds']),'selected_360min_spacing_diagnostic':score(audit_samples,final,chosen['thresholds'],True)};assert abs(reports['round7_frozen']['proxy_IoU']-json.loads((R7/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_groups=sum(r['sigma'] is not None for r in routes.values()),seconds=time.time()-started,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);pr=chosen['alpha_soft']*aa[0]+(1-chosen['alpha_soft'])*bb[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(pr[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',proxy=reports['selected']['proxy_IoU'],new_groups=sum(r['sigma'] is not None for r in routes.values()),seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
