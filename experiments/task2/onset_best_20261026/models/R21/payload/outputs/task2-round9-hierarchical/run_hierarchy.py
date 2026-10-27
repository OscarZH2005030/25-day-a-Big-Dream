"""层级共享：每条件一个全道路父模型，再按道路追加局部树；固定555维+道路one-hot。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';WORK=ROOT.parents[1]/'work/round9-hierarchy';WORK.mkdir(parents=True,exist_ok=True);sys.path[1:1]=[str(R8),str(R7),str(R6),str(R5),str(R2)]
from soft_runtime import SoftPredictor
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
 r,p=extras(s);base=np.concatenate([features(s),r,p],axis=1);onehot=np.zeros((len(base),8),dtype='float32');onehot[:,PANELS.index(s['panel'])]=1;return np.concatenate([base,onehot],axis=1)
def dataset(c):
 folder=WORK/str(c);folder.mkdir(exist_ok=True);done=folder/'complete.json'
 if done.exists():
  meta=json.loads(done.read_text());return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w']},meta
 lengths={p:len(np.load(ROOT.parents[1]/f'work/round5-distribution/{p}-{c}/y.npy',mmap_mode='r')) for p in PANELS};total=sum(lengths.values());xx=np.lib.format.open_memmap(folder/'X.npy',mode='w+',dtype='float32',shape=(total,563));yy=np.lib.format.open_memmap(folder/'y.npy',mode='w+',dtype='float32',shape=(total,));ww=np.lib.format.open_memmap(folder/'w.npy',mode='w+',dtype='float32',shape=(total,));cursor=0;offsets={}
 for pi,p in enumerate(PANELS):
  base=ROOT.parents[1]/f'work/round5-distribution/{p}-{c}';ramps=ROOT.parents[1]/f'work/round6-ramp-matrix/{p}-{c}';x=np.load(base/'X.npy',mmap_mode='r');r=np.load(ramps/'ramp.npy',mmap_mode='r');ph=np.load(ramps/'physics.npy',mmap_mode='r');n=lengths[p];xx[cursor:cursor+n,:555]=np.concatenate([x,r,ph],axis=1);xx[cursor:cursor+n,555:]=0;xx[cursor:cursor+n,555+pi]=1;yy[cursor:cursor+n]=np.load(base/'y.npy',mmap_mode='r');ww[cursor:cursor+n]=np.load(base/'w.npy',mmap_mode='r');offsets[p]=[cursor,cursor+n];cursor+=n;progress('assembling_pooled_data',condition=c,panel=p,rows=cursor,total=total)
 xx.flush();yy.flush();ww.flush();meta=dict(condition=c,rows=total,features=563,offsets=offsets,source='unchanged round5/6 cell IDs, labels and IPW weights; appended observed road identity');atomic_json(done,meta);return dict(X=xx,y=yy,w=ww),meta
def fit(name,X,y,w,total,initial=None,leaves=63):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else initial;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%100==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=total)
 model=lgb.train(dict(objective='binary',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=total-done,init_model=initial,callbacks=[callback]) if done<total else initial;model.save_model(str(path));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps) if s['has_future_queue']]))
def main():
 started=time.time();assert (R8/'inference-reproduction.json').exists(),'等待第八轮独立验收';before=guard('before');baseline=SoftPredictor(R8);bt=baseline.rule['thresholds'];entries=json.loads((R2/'expanded-fit.json').read_text())['samples'];tune=role_samples('tune');pooled={};local={};routes={};development=[];metadata=[]
 save('experiment.json',dict(seed=SEED,parent_models=2,parent_rounds=800,road_finetunes=16,additional_rounds_per_road=200,total_new_tree_rounds=4800,parent='shared across eight panels within history condition, with road one-hot',local='append 200 trees to parent with only this road/condition data',inputs='same 555D causal inputs + eight road identity indicators',supervision='same hard labels, selected rows and IPW weights as round5/6',cpu_threads=4,GPU_training=False,Task1='original 92-file package frozen; historical .942828633 not remeasured',selection='December per-road and temporal half guard; January global mix/threshold; February reused report only',official_S_queue=None))
 causal=[]
 for p in PANELS:
  for c in [0,1]:
   e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
   for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
   v['eligible'][slot+1:]=False;s2=make_sample(v,s['network'],slot,p);assert np.array_equal(matrix(s),matrix(s2));causal.append(dict(panel=p,condition=c,passed=True))
 save('causality-check.json',dict(passed=True,groups=causal,road_indicator_is_observed_identity=True,no_future_gate=True))
 for c in [0,1]:
  data,meta=dataset(c);metadata.append(meta);save('dataset-audit.json',metadata);progress('training_parent',condition=c,rows=meta['rows']);pooled[c]=fit(f'parent-{c}',data['X'],data['y'],data['w'],800)
  for p in PANELS:
   lo,hi=meta['offsets'][p];progress('fine_tuning_road',panel=p,condition=c,rows=hi-lo);local[(p,c)]=fit(f'{p}-{c}-warm',data['X'][lo:hi],data['y'][lo:hi],data['w'][lo:hi],1000,pooled[c],31 if c==0 else 63);ss=[s for s in tune if s['panel']==p and s['condition']==c];xx=[matrix(s) for s in ss];base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[c]);half=[[i for i,s in enumerate(ss) if (pd.Timestamp(s['origin']).day<=15)==first] for first in [True,False]];candidates=[dict(iteration=None,alpha_new=0.,proxy=bv,half_gains=[0.,0.],eligible=True)]
   for iteration in [800,900,1000]:
    raw=[local[(p,c)].predict(x,num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1) for x in xx]
    for alpha in [.25,.5,1.]:
     ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];value=metric(ss,ps,bt[c]);gains=[metric([ss[i] for i in ii],[ps[i] for i in ii],bt[c])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[c]) for ii in half];candidates.append(dict(iteration=iteration,alpha_new=alpha,proxy=value,half_gains=gains,eligible=value>=bv+.003 and min(gains)>=-.02))
   chosen=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[(p,c)]=chosen;development.append(dict(panel=p,condition=c,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',development);save('routes.json',[dict(panel=p,condition=c,**r) for (p,c),r in routes.items()]);del xx;gc.collect()
  del data;gc.collect()
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);r=routes[(s['panel'],s['condition'])];base.append(b)
   if r['iteration'] is None:adapt.append(b)
   else:
    q=local[(s['panel'],s['condition'])].predict(matrix(s),num_iteration=r['iteration'],num_threads=4).astype('float32').reshape(6,-1);adapt.append(r['alpha_new']*q+(1-r['alpha_new'])*b)
  return base,adapt
 progress('calibration');cal=role_samples('calibrate');base,adapt=components(cal);candidates=[dict(alpha_hierarchy=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.25,.5,.75,1.]:
  ps=[alpha*a+(1-alpha)*b for a,b in zip(adapt,base)]
  for t0 in [.075,.1,.15,.2,.25,.35,.5]:
   for t1 in [.075,.1,.15,.2,.25,.35,.5]:candidates.append(dict(alpha_hierarchy=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((R8/'frozen-calibration.json').read_bytes()).hexdigest(),february='reused report only',Task1_frozen=True))
 progress('evaluating');audit=role_samples('audit');base,adapt=components(audit);final=[chosen['alpha_hierarchy']*a+(1-chosen['alpha_hierarchy'])*b for a,b in zip(adapt,base)];reports={'round8_frozen':score(audit,base,bt),'hierarchy_same_thresholds':score(audit,adapt,bt),'selected':score(audit,final,chosen['thresholds']),'selected_360min_spacing_diagnostic':score(audit,final,chosen['thresholds'],True)};assert abs(reports['round8_frozen']['proxy_IoU']-json.loads((R8/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_groups=sum(r['iteration'] is not None for r in routes.values()),seconds=time.time()-started,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=chosen['alpha_hierarchy']*aa[0]+(1-chosen['alpha_hierarchy'])*bb[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',proxy=reports['selected']['proxy_IoU'],new_groups=sum(r['iteration'] is not None for r in routes.values()),seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
