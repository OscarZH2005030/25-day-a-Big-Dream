"""历史匝道与部分物理平衡消融；与第五轮完全相同的监督cell和IPW权重。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';WORK=ROOT.parents[1]/'work/round6-ramp-matrix';WORK.mkdir(parents=True,exist_ok=True);sys.path.insert(1,str(R5));sys.path.insert(2,str(R2))
from run_round2 import sample,role_samples,score,arrays,DATA,PANELS
from distribution_runtime import DistributionPredictor
from special_features import features,latest_state
from ramp_features import extras,day,topology
from task2q.data import make_sample,released_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.baselines import official_persistence,persistence
from task2q.submission import validate_queue
SEED=20261027
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def guard(stage):
 m=json.loads((TASK1/'package-manifest.json').read_text());hashes={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(hashes[n]==e['sha256'] for n,e in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=len(hashes),hashes=hashes,historical_common55_S_state=.942828633,new_Task1_score_measured=False));return hashes
def fit(name,X,y,w,leaves):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%200==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=600)
 model=lgb.train(dict(objective='binary',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=600-done,init_model=initial,callbacks=[callback]) if done<600 else initial;model.save_model(str(path));return model
def group_data(entries,p,c):
 base=ROOT.parents[1]/'work/round5-distribution'/f'{p}-{c}';bm=json.loads((base/'complete.json').read_text());data={k:np.load(base/(k+'.npy'),mmap_mode='r') for k in ['X','y','w']};target=WORK/f'{p}-{c}';target.mkdir(exist_ok=True);sentinel=target/'complete.json'
 if sentinel.exists():return data,np.load(target/'ramp.npy',mmap_mode='r'),np.load(target/'physics.npy',mmap_mode='r'),json.loads(sentinel.read_text())
 selected=[e for e in entries if e['panel']==p and e['condition']==c];rng=np.random.default_rng(SEED+PANELS.index(p)*2+c);rr=[];pp=[];cursor=0
 for e in selected:
  s=sample(e);n=s['x'].shape[1];mask=s['m'].ravel();yy=s['y'].ravel();last,queued=latest_state(s);flip=(yy.astype(bool)!=np.tile(queued,6));mandatory=(mask&(yy>0))|(mask&flip if c==1 else False);keep=np.flatnonzero(mandatory);negative=np.flatnonzero(mask&~mandatory);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.20),6);nr=negative[risk[negative]];no=negative[~risk[negative]];kr=min(96,len(nr));ko=min(192-kr,len(no));kr=min(192-ko,len(nr));ri=rng.choice(nr,kr,replace=False) if kr else np.array([],int);oi=rng.choice(no,ko,replace=False) if ko else np.array([],int);ids=np.concatenate([keep,ri,oi]);k=len(ids);assert np.array_equal(data['y'][cursor:cursor+k],yy[ids]);assert np.array_equal(data['X'][cursor:cursor+min(k,4)],features(s)[ids[:min(k,4)]]);r,ph=extras(s);rr.append(r[ids]);pp.append(ph[ids]);cursor+=k
 assert cursor==len(data['y']);r=np.concatenate(rr);ph=np.concatenate(pp);assert np.isfinite(r).all() and np.isfinite(ph).all();np.save(target/'ramp.npy',r);np.save(target/'physics.npy',ph);meta=dict(panel=p,condition=c,rows=cursor,base_features=data['X'].shape[1],ramp_features=r.shape[1],physics_features=ph.shape[1],same_sample_ids_and_labels_verified=True,IPW_weights_reused=True,history_cutoff='T-5',source='public ramp_states and released topology only',external_CONNECTOR_flows='unknown, explicitly flagged; balance is partial proxy');atomic_json(sentinel,meta);return data,r,ph,meta
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps) if s['has_future_queue']]))
def matrix(s,variant):
 r,p=extras(s);return np.concatenate([features(s),r]+([p] if variant=='physics' else []),axis=1)
def main():
 started=time.time();before=guard('before');assert (R5/'final-verification.json').exists(),'第五轮尚未完成';baseline=DistributionPredictor(R5);bt=baseline.rule['thresholds'];entries=json.loads((R2/'expanded-fit.json').read_text())['samples'];tune=role_samples('tune');models={};routes={};development=[];metadata=[]
 save('experiment.json',dict(seed=SEED,models=32,rounds_per_model=600,max_total_tree_rounds=19200,cpu_threads=4,GPU_training=False,cpu_temperature=None,inputs='490D base; +44D strictly T-60..T-5 public ramp history; +21D internal-flow/partial-balance/topological-position',ablation='same round5 sample IDs, labels, IPW weights, tree parameters and seed; compare ramp-only vs ramp+partial-physics',Task1='92-file frozen package guard; historical .942828633, not remeasured',selection='December per-road/condition and half-month stability guard; January global calibration; February reused report only',official_S_queue=None))
 causal=[]
 for p in PANELS:
  for c in [0,1]:
   e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);origin=pd.Timestamp(s['origin']);date=origin.strftime('%Y-%m-%d');a=day(p,'train',date).copy();slot=origin.hour*12+origin.minute//5;a[slot:]=999;r1,p1=extras(s);r2,p2=extras(s,{date:a});assert np.array_equal(r1,r2) and np.array_equal(p1,p2);v=copy.deepcopy(arrays(e['day_key']))
   for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
   v['eligible'][slot+1:]=False;s2=make_sample(v,s['network'],slot,p);assert np.array_equal(matrix(s,'physics'),matrix(s2,'physics'));causal.append(dict(panel=p,condition=c,passed=True,ramp_T_and_future_perturbed=True,mainline_future_perturbed=True))
 save('causality-check.json',dict(passed=True,groups=causal,ramps_cutoff='T-5',external_boundary_flows_not_fabricated=True))
 for p in PANELS:
  for c in [0,1]:
   tag=f'{p}-{c}';progress('building_ramp_inputs',panel=p,condition=c,completed=len(routes),total=16);data,r,ph,meta=group_data(entries,p,c);metadata.append(meta);save('input-audit.json',metadata)
   for variant in ['ramp','physics']:
    xx=np.concatenate([data['X'],r]+([ph] if variant=='physics' else []),axis=1);models[tag+'-'+variant]=fit(tag+'-'+variant,xx,data['y'],data['w'],31 if c==0 else 63);del xx;gc.collect()
   del data,r,ph;gc.collect();ss=[s for s in tune if s['panel']==p and s['condition']==c];base=[baseline.probability(s) for s in ss];threshold=bt[c];bv=metric(ss,base,threshold);half=[[i for i,s in enumerate(ss) if (pd.Timestamp(s['origin']).day<=15)==first] for first in [True,False]];candidates=[dict(variant='baseline',iteration=None,alpha_new=0.,proxy=bv,half_gains=[0.,0.],eligible=True)]
   for variant in ['ramp','physics']:
    matrices=[matrix(s,variant) for s in ss]
    for iteration in [200,400,600]:
     raw=[models[tag+'-'+variant].predict(x,num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1) for x in matrices]
     for alpha in [.25,.5,1.]:
      ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];value=metric(ss,ps,threshold);gains=[metric([ss[i] for i in ii],[ps[i] for i in ii],threshold)-metric([ss[i] for i in ii],[base[i] for i in ii],threshold) for ii in half];candidates.append(dict(variant=variant,iteration=iteration,alpha_new=alpha,proxy=value,half_gains=gains,eligible=value>=bv+.003 and min(gains)>=-.02))
    del matrices
   chosen=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[(p,c)]=chosen;development.append(dict(panel=p,condition=c,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',development);save('routes.json',[dict(panel=p,condition=c,**r) for (p,c),r in routes.items()])
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);r=routes[(s['panel'],s['condition'])];base.append(b)
   if r['variant']=='baseline':adapt.append(b)
   else:
    raw=models[f"{s['panel']}-{s['condition']}-{r['variant']}"].predict(matrix(s,r['variant']),num_iteration=r['iteration'],num_threads=4).astype('float32').reshape(6,-1);adapt.append(r['alpha_new']*raw+(1-r['alpha_new'])*b)
  return base,adapt
 progress('calibration');cal=role_samples('calibrate');base,adapt=components(cal);candidates=[dict(alpha_ramp=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.25,.5,.75,1.]:
  ps=[alpha*a+(1-alpha)*b for a,b in zip(adapt,base)]
  for t0 in [.05,.075,.1,.15,.2,.25,.35]:
   for t1 in [.075,.1,.15,.2,.25,.35,.5]:candidates.append(dict(alpha_ramp=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_bundle=str(R5),baseline_frozen_sha256=hashlib.sha256((R5/'frozen-calibration.json').read_bytes()).hexdigest(),february='reused report only',Task1_frozen=True))
 progress('evaluating');audit=role_samples('audit');base,adapt=components(audit);final=[chosen['alpha_ramp']*a+(1-chosen['alpha_ramp'])*b for a,b in zip(adapt,base)];reports={'official_persistence':score(audit,[official_persistence(s,s['network']) for s in audit]),'carry_T':score(audit,[persistence(s,s['network'],True) for s in audit]),'round5_frozen':score(audit,base,bt),'ramp_same_thresholds':score(audit,adapt,bt),'selected':score(audit,final,chosen['thresholds']),'selected_360min_spacing_diagnostic':score(audit,final,chosen['thresholds'],True)};assert abs(reports['round5_frozen']['proxy_IoU']-json.loads((R5/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_groups=sum(r['variant']!='baseline' for r in routes.values()),seconds=time.time()-started,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);pr=chosen['alpha_ramp']*aa[0]+(1-chosen['alpha_ramp'])*bb[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(pr[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,causal_checks_passed=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',proxy=reports['selected']['proxy_IoU'],new_groups=sum(r['variant']!='baseline' for r in routes.values()),seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
