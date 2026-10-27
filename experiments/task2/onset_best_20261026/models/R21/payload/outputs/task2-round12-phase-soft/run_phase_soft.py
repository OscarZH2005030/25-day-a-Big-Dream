"""组合起点覆盖与软目标，仅改Onset；Ongoing预测逐位冻结。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,hashlib,traceback,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R11=ROOT.parent/'task2-round11-phase-validation';R10=ROOT.parent/'task2-round10-phase-coverage';R9=ROOT.parent/'task2-round9-hierarchical';R8=ROOT.parent/'task2-round8-soft-targets';R7=ROOT.parent/'task2-round7-structured';R6=ROOT.parent/'task2-round6-ramp-physics';R5=ROOT.parent/'task2-round5-distribution';R2=ROOT.parent/'task2-round2';WORK=ROOT.parents[1]/'work/round12-phase-soft';WORK.mkdir(parents=True,exist_ok=True);sys.path[1:1]=[str(R11),str(R10),str(R9),str(R8),str(R7),str(R6),str(R5),str(R2)]
from phase_runtime import PhasePredictor
from coverage_data import matrix
from run_round2 import sample,role_samples,score,DATA,PANELS
from special_features import latest_state
from task2q.cache import atomic_json
from task2q.data import released_sample
from task2q.metrics import window_iou
from task2q.submission import validate_queue
SEED=20261027;TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def guard(stage):
 m=read(TASK1/'package-manifest.json');h={n:hashlib.sha256((TASK1/n).read_bytes()).hexdigest() for n in m['files']};assert all(h[n]==v['sha256'] for n,v in m['files'].items());save('task1-guard-'+stage+'.json',dict(passed=True,files=92,hashes=h,historical_S_state=.942828633,new_Task1_score_measured=False));return h
def margin_data(entries,p):
 folder=ROOT.parents[1]/f'work/round10-phase/{p}-0';data={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w']};meta=read(folder/'complete.json');cache=WORK/(p+'-margin.npy');auditfile=WORK/(p+'-audit.json')
 if cache.exists() and auditfile.exists():return data,np.load(cache,mmap_mode='r'),read(auditfile)
 margin=np.zeros(len(data['y']),dtype='float32');old=meta['original_rows'];x=data['X'];residual=np.load(ROOT.parents[1]/f'work/round5-distribution/{p}-0/residual.npy',mmap_mode='r');assert len(residual)==old;margin[:old]=(x[:old,173]-(residual+x[:old,175]))*(x[:old,171]*130);lookup={e['origin']:e for e in entries if e['panel']==p and e['condition']==0 and e['source']=='added_phase'};rng=np.random.default_rng(SEED+2000+PANELS.index(p)*2)
 for window in meta['windows']:
  if window['source']=='original':continue
  e=lookup[window['origin']];s=sample(e);n=s['x'].shape[1];m=s['m'].ravel();y=s['y'].ravel();last,_=latest_state(s);mandatory=m&(y>0);keep=np.flatnonzero(mandatory);negative=np.flatnonzero(m&~mandatory);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.20),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);lo=window['start'];hi=window['end'];assert hi-lo==len(ids) and np.array_equal(data['y'][lo:hi],y[ids]);nodes=ids%n;margin[lo:hi]=s['network']['threshold'][nodes]-s['speed_y'].ravel()[ids]*s['network']['free_speed'][nodes]
 away=np.abs(margin)>.001;assert np.array_equal((margin[away]>=0).astype('float32'),data['y'][away]);np.save(cache,margin);audit=dict(panel=p,rows=len(margin),original_rows=old,added_rows=len(margin)-old,expanded_windows=meta['expanded_windows'],features=555,sign_matches_hard_labels=True,added_cell_ids_replayed_exactly=True,near_threshold_5kmh_fraction=float((np.abs(margin)<5).mean()));atomic_json(auditfile,audit);return data,margin,audit
def fit(name,X,y,w):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%200==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=600)
 model=lgb.train(dict(objective='cross_entropy',num_leaves=31,learning_rate=.035,min_data_in_leaf=80,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True),lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=600-done,init_model=initial,callbacks=[callback]) if done<600 else initial;model.save_model(str(path));return model
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps)]))
def main():
 started=time.time();assert (R11/'inference-reproduction.json').exists();before=guard('before');baseline=PhasePredictor(R11);bt=baseline.rule['thresholds'];entries=read(R10/'expanded-fit.json')['samples'];tune=[sample(e) for e in read(R11/'tune-phase-index.json')['samples']];hold={(r['panel'],r['condition']):r['deployment_guard_passed'] for r in read(R10/'temporal-holdout.json')};models={};routes={};development=[];audit=[]
 save('experiment.json',dict(seed=SEED,models=16,rounds_per_model=600,max_tree_rounds=9600,new_hypothesis='factorial combination of real origin-phase coverage and soft noisy threshold targets; onset only',inputs='same555D and exact expanded round10 row IDs/IPW weights',sigmas_kmh=[2.5,5.],objective='cross_entropy',ongoing='entire probability path and decision threshold fixed to round11',selection='December phase-balanced population, min4 windows per phase, temporal holdout guard, gain>=.005 and worst phase>=-.02',calibration='January global onset mix/threshold only; ongoing frozen',audit='same new-phase and old410 populations as round11, both repeated development-after-evaluation',cpu_threads=4,GPU_training=False,Task1='92file frozen package, historical .942828633 not remeasured',official_S_queue=None))
 for p in PANELS:
  progress('preparing_expanded_soft_targets',panel=p,completed=len(routes),total=8);data,margin,info=margin_data(entries,p);audit.append(info);save('soft-label-coverage-audit.json',audit)
  for sigma in [2.5,5.]:soft=(1/(1+np.exp(np.clip(-margin/sigma,-30,30)))).astype('float32');models[(p,sigma)]=fit(f'{p}-sigma{sigma}',data['X'],soft,data['w'])
  del data,margin;gc.collect();ss=[s for s in tune if s['panel']==p and s['condition']==0];xx=[matrix(s) for s in ss];phase=np.array([(pd.Timestamp(s['origin']).hour*12+pd.Timestamp(s['origin']).minute//5)%3 for s in ss]);base=[baseline.probability(s) for s in ss];bv=metric(ss,base,bt[0]);candidates=[dict(sigma=None,alpha_new=0.,proxy=bv,phase_gains=[0.,0.,0.],eligible=True)]
  for sigma in [2.5,5.]:
   raw=[models[(p,sigma)].predict(x,num_threads=4).astype('float32').reshape(6,-1) for x in xx]
   for alpha in [.25,.5,1.]:
    ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];gains=[];counts=[]
    for ph in [0,1,2]:
     ii=np.flatnonzero(phase==ph);counts.append(len(ii));gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],bt[0])-metric([ss[i] for i in ii],[base[i] for i in ii],bt[0]) if len(ii) else -1.)
    value=metric(ss,ps,bt[0]);candidates.append(dict(sigma=sigma,alpha_new=alpha,proxy=value,phase_gains=gains,eligible=hold[(p,0)] and min(counts)>=4 and value>=bv+.005 and min(gains)>=-.02))
  chosen=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[p]=chosen;development.append(dict(panel=p,condition=0,baseline_proxy=bv,selected=chosen,candidates=candidates));save('december-selection.json',development);save('routes.json',[dict(panel=p,**r) for p,r in routes.items()]);del xx;gc.collect()
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);base.append(b)
   if s['condition']==1 or routes[s['panel']]['sigma'] is None:adapt.append(b)
   else:
    r=routes[s['panel']];q=models[(s['panel'],r['sigma'])].predict(matrix(s),num_threads=4).astype('float32').reshape(6,-1);adapt.append(r['alpha_new']*q+(1-r['alpha_new'])*b)
  return base,adapt
 def mixed(ss,base,adapt,alpha):return [b if s['condition']==1 else alpha*a+(1-alpha)*b for s,a,b in zip(ss,adapt,base)]
 progress('calibration');cal=[sample(e) for e in read(R11/'calibrate-phase-index.json')['samples']];base,adapt=components(cal);candidates=[dict(alpha_phase_soft=0.,thresholds=bt,proxy=score(cal,base,bt)['proxy_IoU'])]
 for alpha in [0.,.25,.5,1.]:
  ps=mixed(cal,base,adapt,alpha)
  for t0 in [.075,.1,.15,.2,.25,.35,.5]:candidates.append(dict(alpha_phase_soft=alpha,thresholds=[t0,bt[1]],proxy=score(cal,ps,[t0,bt[1]])['proxy_IoU']))
 chosen=max(candidates,key=lambda a:a['proxy']);hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.txt')};save('frozen-calibration.json',dict(selected=chosen,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),baseline_frozen_sha256=hashlib.sha256((R11/'frozen-calibration.json').read_bytes()).hexdigest(),ongoing_threshold_frozen=bt[1],February_previously_used_report_only=True,Task1_frozen=True))
 progress('evaluating');window_records=[];reports={};ongoing_identical=True
 for population,ss in [('new_phase_population',[sample(e) for e in read(R11/'audit-phase-index.json')['samples']]),('old410',role_samples('audit'))]:
  base,adapt=components(ss);final=mixed(ss,base,adapt,chosen['alpha_phase_soft']);reports[population+'_round11']=score(ss,base,bt);reports[population+'_selected']=score(ss,final,chosen['thresholds'])
  for s,b,q in zip(ss,base,final):
   if not s['has_future_queue']:continue
   if s['condition']==1:assert np.array_equal(b>=bt[1],q>=chosen['thresholds'][1])
   window_records.append(dict(population=population,panel=s['panel'],condition=s['condition'],origin=s['origin'],date=s['origin'][:10],base_iou=window_iou(s['y'],b>=bt[s['condition']],s['m']),selected_iou=window_iou(s['y'],q>=chosen['thresholds'][s['condition']],s['m'])))
  expected=read(R11/'proxy-results.json')['reports'][population+'_selected']['proxy_IoU'];assert abs(reports[population+'_round11']['proxy_IoU']-expected)<1e-12
 save('paired-window-metrics.json',window_records);save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=chosen,new_onset_roads=sum(r['sigma'] is not None for r in routes.values()),seconds=time.time()-started,ongoing_predictions_identical=True,different_populations_not_directly_comparable=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   d=DATA/'task2'/p/split;index=pd.read_csv(d/'window_index.csv');template=pd.read_csv(d/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);q=mixed([s],bb,aa,chosen['alpha_phase_soft'])[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(q[h,keys.link_id.map(ii).to_numpy()]>=chosen['thresholds'][s['condition']]).astype(int);parts.append(keys)
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
