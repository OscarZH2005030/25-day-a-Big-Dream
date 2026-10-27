"""固定硬目标/覆盖/参数，只改随机种子；Onset三成员集成，冻结Ongoing。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,gc,re
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R12=ROOT.parent/'task2-round12-phase-soft';R11=ROOT.parent/'task2-round11-phase-validation';R10=ROOT.parent/'task2-round10-phase-coverage';R9=ROOT.parent/'task2-round9-hierarchical';R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';sys.path[1:1]=[str(R12),str(R11),str(R10),str(R9),str(R8),str(R7),str(R6),str(R5),str(R2)]
from phase_soft_runtime import PhaseSoftPredictor
from coverage_data import matrix
from run_round2 import sample,role_samples,score,DATA,PANELS
from task2q.cache import atomic_json
from task2q.data import released_sample
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEEDS=[20261027,20261028,20261029];TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def guard(stage):
 m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=92,hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def fit(name,X,y,w,seed):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%200==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=600)
 model=lgb.train(dict(objective='binary',num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=seed,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=600-done,init_model=initial,callbacks=[callback]) if done<600 else initial;model.save_model(str(path));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def main():
 started=time.time();assert (R12/'inference-reproduction.json').exists();before=guard('before');baseline=PhaseSoftPredictor(R12);bt=baseline.rule['thresholds'];tune=[sample(e) for e in read(R11/'tune-phase-index.json')['samples']];hold={(r['panel'],r['condition']):r['deployment_guard_passed'] for r in read(R10/'temporal-holdout.json')};models={};routes={};development=[];stability=[];seed_params=[]
 save('experiment.json',dict(seeds=SEEDS,seed0='unchanged round10 complete600 hard-target model',new_models=16,rounds_per_model=600,new_tree_rounds=9600,objective='binary hard queue; NOT tuned sigma; data/labels/IPW/555features/hyperparams fixed',ensemble='equal mean of three seeds, no performance-weighted members',selection='December phase-balanced onset only with temporal guard; January coarse onset-only calibration',ongoing='all probabilities and threshold frozen to round12',cpu_threads=4,GPU_training=False,Task1='92file frozen package historical .942828633 not remeasured',official_S_queue=None))
 for p in PANELS:
  progress('training_seed_group',panel=p,completed=len(routes),total=8);folder=ROOT.parents[1]/f'work/round10-phase/{p}-0';data={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w']};models[(p,SEEDS[0])]=lgb.Booster(model_file=str(R10/f'{p}-0-expanded.txt'))
  for seed in SEEDS[1:]:models[(p,seed)]=fit(f'{p}-seed{seed}',data['X'],data['y'],data['w'],seed)
  del data;gc.collect()
  for seed in SEEDS:
   path=R10/f'{p}-0-expanded.txt' if seed==SEEDS[0] else ROOT/f'{p}-seed{seed}.txt';text=path.read_text(encoding='utf8');params={k:int(re.search(r'\['+k+r': (\d+)\]',text).group(1)) for k in ['seed','feature_fraction_seed','data_random_seed']};seed_params.append(dict(panel=p,**params))
  save('seed-stream-audit.json',seed_params);ss=[s for s in tune if s['panel']==p and s['condition']==0];xx=[matrix(s) for s in ss];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);raw={seed:[models[(p,seed)].predict(x,num_threads=4).astype('float32').reshape(6,-1) for x in xx] for seed in SEEDS};ens=[np.mean(np.stack([raw[seed][i] for seed in SEEDS]),axis=0).astype('float32') for i in range(len(ss))];base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[0]);candidate=[dict(alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)];sigmas=[];disagreements=[]
  for i,s in enumerate(ss):
   preds=np.stack([raw[seed][i] for seed in SEEDS]);sigmas.extend(preds.std(0)[s['m']].tolist());decisions=preds>=bt[0];disagreements.extend((decisions.any(0)!=decisions.all(0))[s['m']].tolist())
  stability.append(dict(panel=p,windows=len(ss),member_seed_IoUs={str(seed):metric(ss,raw[seed],bt[0]) for seed in SEEDS},ensemble_IoU=metric(ss,ens,bt[0]),mean_eligible_probability_std=float(np.mean(sigmas)),eligible_decision_disagreement_fraction=float(np.mean(disagreements))));save('december-seed-stability.json',stability)
  for alpha in [.25,.5,1.]:
   ps=[alpha*a+(1-alpha)*b for a,b in zip(ens,base)];gains=[];counts=[]
   for ph in [0,1,2]:
    ii=np.flatnonzero(phase==ph);counts.append(len(ii));gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[0])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[0]) if len(ii) else -1.)
   value=metric(ss,ps,bt[0]);candidate.append(dict(alpha_new=alpha,proxy=value,phase_gains=gains,eligible=hold[(p,0)] and min(counts)>=4 and value>=bv+.003 and min(gains)>=-.015))
  chosen=max((a for a in candidate if a['eligible']),key=lambda a:a['proxy']);routes[p]=chosen;development.append(dict(panel=p,condition=0,baseline_proxy=bv,selected=chosen,candidates=candidate));save('december-selection.json',development);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()]);del xx,raw,ens;gc.collect()
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);base.append(b)
   if s['condition']==1 or routes[s['panel']]['alpha_new']==0:adapt.append(b)
   else:
    x=matrix(s);q=np.mean(np.stack([models[(s['panel'],seed)].predict(x,num_threads=4).astype('float32').reshape(6,-1) for seed in SEEDS]),axis=0).astype('float32');a=routes[s['panel']]['alpha_new'];adapt.append(a*q+(1-a)*b)
  return base,adapt
 def mixed(ss,b,a,alpha):return [bb if s['condition']==1 else alpha*aa+(1-alpha)*bb for s,aa,bb in zip(ss,a,b)]
 progress('calibration');cal=[sample(e) for e in read(R11/'calibrate-phase-index.json')['samples']];base,adapt=components(cal);candidates=[dict(alpha_seed=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.25,.5,1.]:
  ps=mixed(cal,base,adapt,alpha)
  for t0 in [.2,.35,.5]:candidates.append(dict(alpha_seed=alpha,thresholds=[t0,bt[1]],proxy=score(cal,ps,[t0,bt[1]])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,seed0_hashes={f'{p}-0-expanded.txt':hashlib.sha256((R10/f'{p}-0-expanded.txt').read_bytes()).hexdigest() for p in PANELS},route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((R12/'frozen-calibration.json').read_bytes()).hexdigest(),ongoing_frozen=True,February_previously_used_report_only=True,Task1_frozen=True))
 progress('evaluating');reports={};window_records=[]
 for population,ss in [('new_phase_population',[sample(e) for e in read(R11/'audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
  base,adapt=components(ss);final=mixed(ss,base,adapt,chosen['alpha_seed']);reports[population+'_round12']=score(ss,base,bt);reports[population+'_selected']=score(ss,final,chosen['thresholds'])
  for s,b,q in zip(ss,base,final):
   if not s['has_future_queue']:continue
   if s['condition']==1:assert np.array_equal(b>=bt[1],q>=chosen['thresholds'][1])
   window_records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],b>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=chosen['thresholds'][s['condition']],s['m'])))
  assert abs(reports[population+'_round12']['proxy_IoU']-read(R12/'proxy-results.json')['reports'][population+'_selected']['proxy_IoU'])<1e-12
 save('paired-window-metrics.json',window_records);save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_onset_roads=sum(r['alpha_new']>0 for r in routes.values()),seconds=time.time()-started,ongoing_predictions_identical=True,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=mixed([s],bb,aa,chosen['alpha_seed'])[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 assert guard('after')==before;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,ongoing_predictions_identical=True,Task1_package_unchanged=True,official_S_queue=None));progress('complete',new_phase_proxy=reports['new_phase_population_selected']['proxy_IoU'],old410_proxy=reports['old410_selected']['proxy_IoU'],seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
