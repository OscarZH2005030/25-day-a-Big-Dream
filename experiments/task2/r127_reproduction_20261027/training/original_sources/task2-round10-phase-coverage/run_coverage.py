"""覆盖扩展与训练内时间留出：保留旧监督，补5/10分钟起点；固定模型参数。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R9=ROOT.parent/'task2-round9-hierarchical';R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';sys.path[1:1]=[str(R9),str(R8),str(R7),str(R6),str(R5),str(R2)]
from hierarchy_runtime import HierarchyPredictor
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from coverage_data import expanded_index,group_dataset,matrix
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
def fit(name,X,y,w,rounds,leaves):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%200==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=rounds)
 model=lgb.train(dict(objective='binary',num_leaves=leaves,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial;model.save_model(str(path));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps) if s['has_future_queue']]))
def main():
 started=time.time();before=guard('before');baseline=HierarchyPredictor(R9);bt=baseline.rule['thresholds'];save('experiment.json',dict(seed=SEED,original_windows=8192,target_windows=16384,new_windows='true public fit origins at phase +5/+10 minutes; original phase0 retained',features=555,full_models=16,full_rounds=600,temporal_probes=32,probe_rounds=200,max_total_tree_rounds=16000,temporal_holdout='June-September train; October-November original positive windows evaluate paired control vs expanded',temporal_guard='median gain over fixed .1/.2/.35 thresholds >=-.01 required before December deployment',selection='December half-month stability; January global calibration; February repeated report only',cpu_threads=4,GPU_training=False,Task1='92-file frozen guard, .942828633 historical only',official_S_queue=None))
 phase={str(i):0 for i in range(3)}
 for p in PANELS:
  f=pd.read_csv(DATA/'task2'/p/'train/window_index.csv');t=pd.to_datetime(f.forecast_origin,utc=True)
  for i,n in ((t.dt.hour*12+t.dt.minute//5)%3).value_counts().items():phase[str(i)]+=int(n)
 save('origin-phase-audit.json',dict(original_fit_mod3={'0':8192,'1':0,'2':0},official_train_mod3=phase,official_train_only=True,no_hidden_labels_used=True))
 index=expanded_index(progress);entries=index['samples'];save('coverage-audit.json',dict(counts=index['counts'],windows=len(entries),original_windows=sum(e['source']=='original' for e in entries),added_windows=sum(e['source']=='added_phase' for e in entries),note='5/10 minute phase and source-time coverage; windows overlap and do not double independent events'))
 causal=[]
 for p in PANELS:
  for c in [0,1]:
   e=next(e for e in entries if e['panel']==p and e['condition']==c and e['source']=='added_phase');s=sample(e);v=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
   for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
   v['eligible'][slot+1:]=False;s2=make_sample(v,s['network'],slot,p);assert np.array_equal(matrix(s),matrix(s2));causal.append(dict(panel=p,condition=c,phase=int(slot%3),passed=True))
 save('causality-check.json',dict(passed=True,groups=causal,new_origin_phase_does_not_expand_history_boundary=True,ramps_remain_Tminus5=True))
 tune=role_samples('tune');models={};routes={};development=[];holdout_reports=[];data_audit=[]
 for p in PANELS:
  for c in [0,1]:
   tag=f'{p}-{c}';progress('building_coverage_rows',panel=p,condition=c,completed=len(routes),total=16);data,meta=group_dataset(entries,p,c,progress);data_audit.append({k:v for k,v in meta.items() if k!='windows'});save('training-data-audit.json',data_audit);hold=[sample(e) for e in entries if e['panel']==p and e['condition']==c and e['source']=='original' and e['origin'][:10]>='2030-10-01' and e['has_future_queue']];hx=[matrix(s) for s in hold];probe_predictions={}
   for probe,key in [('control','control_early'),('expanded','early')]:
    selected=data[key].astype(bool);windows=[r for r in meta['windows'] if r['early'] and (probe!='control' or r['source']=='original')];counts={b:sum(r['positive']==b for r in windows) for b in [False,True]};allcounts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]};weights=np.asarray(data['w'][selected])*np.where(data['window_positive'][selected]>.5,allcounts[True]/counts[True],allcounts[False]/counts[False]);model=fit(tag+'-probe-'+probe,np.asarray(data['X'][selected]),np.asarray(data['y'][selected]),weights,200,31);probe_predictions[probe]=[model.predict(x,num_threads=4).astype('float32').reshape(6,-1) for x in hx]
   hr=[dict(threshold=t,control_IoU=metric(hold,probe_predictions['control'],t),expanded_IoU=metric(hold,probe_predictions['expanded'],t)) for t in [.1,.2,.35]];gains=[a['expanded_IoU']-a['control_IoU'] for a in hr];temporal_ok=bool(np.median(gains)>=-.01);holdout_reports.append(dict(panel=p,condition=c,windows=len(hold),reports=hr,median_gain=float(np.median(gains)),deployment_guard_passed=temporal_ok));save('temporal-holdout.json',holdout_reports);del hx,probe_predictions;gc.collect();models[(p,c)]=fit(tag+'-expanded',data['X'],data['y'],data['w'],600,31 if c==0 else 63);del data;gc.collect()
   ss=[s for s in tune if s['panel']==p and s['condition']==c];xx=[matrix(s) for s in ss];base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[c]);half=[[i for i,s in enumerate(ss) if (pd.Timestamp(s['origin']).day<=15)==first] for first in [True,False]];candidates=[dict(iteration=None,alpha_new=0.,proxy=bv,half_gains=[0.,0.],eligible=True)]
   for iteration in [200,400,600]:
    raw=[models[(p,c)].predict(x,num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1) for x in xx]
    for alpha in [.25,.5,1.]:
     ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];value=metric(ss,ps,bt[c]);gain=[metric([ss[i] for i in ii],[ps[i] for i in ii],bt[c])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[c]) for ii in half];candidates.append(dict(iteration=iteration,alpha_new=alpha,proxy=value,half_gains=gain,eligible=temporal_ok and value>=bv+.003 and min(gain)>=-.02))
   chosen=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[(p,c)]=chosen;development.append(dict(panel=p,condition=c,temporal_guard=temporal_ok,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',development);save('routes.json',[dict(panel=p,condition=c,**r) for (p,c),r in routes.items()]);del xx;gc.collect()
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);r=routes[(s['panel'],s['condition'])];base.append(b)
   if r['iteration'] is None:adapt.append(b)
   else:
    q=models[(s['panel'],s['condition'])].predict(matrix(s),num_iteration=r['iteration'],num_threads=4).astype('float32').reshape(6,-1);adapt.append(r['alpha_new']*q+(1-r['alpha_new'])*b)
  return base,adapt
 progress('calibration');cal=role_samples('calibrate');base,adapt=components(cal);candidates=[dict(alpha_coverage=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.25,.5,.75,1.]:
  ps=[alpha*a+(1-alpha)*b for a,b in zip(adapt,base)]
  for t0 in [.075,.1,.15,.2,.25,.35,.5]:
   for t1 in [.075,.1,.15,.2,.25,.35,.5]:candidates.append(dict(alpha_coverage=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((R9/'frozen-calibration.json').read_bytes()).hexdigest(),february='reused report only',Task1_frozen=True))
 progress('evaluating');audit=role_samples('audit');base,adapt=components(audit);final=[chosen['alpha_coverage']*a+(1-chosen['alpha_coverage'])*b for a,b in zip(adapt,base)];reports={'round9_frozen':score(audit,base,bt),'coverage_same_thresholds':score(audit,adapt,bt),'selected':score(audit,final,chosen['thresholds']),'selected_360min_spacing_diagnostic':score(audit,final,chosen['thresholds'],True)};assert abs(reports['round9_frozen']['proxy_IoU']-json.loads((R9/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_groups=sum(r['iteration'] is not None for r in routes.values()),seconds=time.time()-started,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=chosen['alpha_coverage']*aa[0]+(1-chosen['alpha_coverage'])*bb[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
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
