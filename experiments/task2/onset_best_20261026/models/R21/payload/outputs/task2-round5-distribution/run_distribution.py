"""分布驱动：Onset事件分解；Ongoing状态转移/连续速度；Task1冻结保护。"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import sys,json,time,copy,hashlib,traceback,gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parent;R2=ROOT.parent/'task2-round2';R4=ROOT.parent/'task2-round4-road-adaptive';WORK=ROOT.parents[1]/'work/round5-distribution';WORK.mkdir(parents=True,exist_ok=True)
sys.path.insert(1,str(R2))
from run_round2 import sample,arrays,role_samples,score,DATA,PANELS
from task2q.data import released_sample,make_sample
from task2q.cache import atomic_json
from task2q.metrics import window_iou
from task2q.baselines import official_persistence,persistence
from task2q.submission import validate_queue
from special_features import features,latest_state
from baseline_runtime import Baseline
SEED=20261027
TASK1=Path(r'C:\Users\18049\Documents\Codex\2026-10-24\trafficflowbench-186-global-road-adaptation\outputs\TrafficFlowBench-Best-20261026')
def save(n,v):atomic_json(ROOT/n,v)
def progress(phase,**kw):
 v=dict(phase=phase,pid=os.getpid(),utc=pd.Timestamp.now(tz='UTC').isoformat(),**kw);save('progress.json',v);print(json.dumps(v,ensure_ascii=True),flush=True)
def task1_guard(stage):
 manifest=json.loads((TASK1/'package-manifest.json').read_text(encoding='utf8'));records={}
 for n,expected in manifest['files'].items():
  p=TASK1/n;h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==expected['sha256'] and p.stat().st_size==expected['bytes'],'Task1 package changed: '+n;records[n]=h
 v=dict(passed=True,stage=stage,files=len(records),hashes=records,historical_common55_S_state=.942828633,new_Task1_score_measured=False,Task1_route_unchanged=True,Task2_does_not_read_Task1_predictions=True,official_S_state=None);save('task1-guard-'+stage+'.json',v);return records
def fit(name,X,y,w,rounds,leaves=31,minleaf=80,objective='binary'):
 path=ROOT/(name+'.txt')
 if path.exists():return lgb.Booster(model_file=str(path))
 checkpoint=ROOT/(name+'-checkpoint.txt');initial=lgb.Booster(model_file=str(checkpoint)) if checkpoint.exists() else None;done=initial.current_iteration() if initial else 0
 def callback(env):
  if (env.iteration+1)%200==0:env.model.save_model(str(checkpoint));progress('training',model=name,step=env.iteration+1,total=rounds)
 params=dict(objective=objective,num_leaves=leaves,learning_rate=.035,min_data_in_leaf=minleaf,feature_fraction=.85,lambda_l2=4,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True)
 if objective=='huber':params['alpha']=.9
 model=lgb.train(params,lgb.Dataset(X,label=y,weight=w/w.mean()),num_boost_round=rounds-done,init_model=initial,callbacks=[callback]) if done<rounds else initial;model.save_model(str(path));return model
def group_data(entries,panel,condition):
 directory=WORK/f'{panel}-{condition}';directory.mkdir(exist_ok=True);sentinel=directory/'complete.json'
 if sentinel.exists():
  meta=json.loads(sentinel.read_text());return {k:np.load(directory/(k+'.npy'),mmap_mode='r') for k in meta['arrays']},meta
 rng=np.random.default_rng(SEED+PANELS.index(panel)*2+condition);selected=[e for e in entries if e['panel']==panel and e['condition']==condition];strata={positive:sum(e['has_future_queue']==positive for e in selected) for positive in [False,True]};chunks={k:[] for k in ['X','y','w','flip','residual']+(['eventX','eventY','eventW','condX','condY','condW'] if condition==0 else [])};full_positive=0;full_transition=0;old_expected_positive=0;windows=0
 for e in selected:
  s=sample(e);xx=features(s);n=s['x'].shape[1];mask=s['m'].ravel();yy=s['y'].ravel();last,queued=latest_state(s);base=np.tile(queued,6);flip=(yy.astype(bool)!=base).astype('float32');residual=s['speed_y'].ravel()-np.tile(last,6);valid=np.flatnonzero(mask);pos=np.flatnonzero(mask&(yy>0));full_positive+=len(pos);old_expected_positive+=96*len(pos)/len(valid);mandatory=(mask&(yy>0))|(mask&(flip>0) if condition==1 else False);full_transition+=int((mask&(flip>0)).sum());keep=np.flatnonzero(mandatory);negative=np.flatnonzero(mask&~mandatory)
  # 困难负例按历史风险分层抽样，保留已知包含概率并用逆概率校正，避免简单过采样改变先验。
  risk_link=(last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.20);risk=np.tile(risk_link,6);neg_risk=negative[risk[negative]];neg_other=negative[~risk[negative]];kr=min(96,len(neg_risk));ko=min(192-kr,len(neg_other));kr=min(192-ko,len(neg_risk));ri=rng.choice(neg_risk,kr,replace=False) if kr else np.array([],dtype=int);oi=rng.choice(neg_other,ko,replace=False) if ko else np.array([],dtype=int);ids=np.concatenate([keep,ri,oi]);inverse=np.concatenate([np.ones(len(keep)),np.full(kr,len(neg_risk)/kr if kr else 0),np.full(ko,len(neg_other)/ko if ko else 0)]);window_weight=(.8 if e['has_future_queue'] else .2)/strata[e['has_future_queue']];weights=inverse/len(valid)*window_weight
  chunks['X'].append(xx[ids]);chunks['y'].append(yy[ids]);chunks['w'].append(weights.astype('float32'));chunks['flip'].append(flip[ids]);chunks['residual'].append(residual[ids]);windows+=1
  if condition==0:
   event=s['y'].astype(bool).any(0);known=event|s['m'].all(0);event_ids=np.flatnonzero(known);ex=np.concatenate([xx[:n,:185],xx[:n,191:]],axis=1)
   if len(event_ids):chunks['eventX'].append(ex[event_ids]);chunks['eventY'].append(event[event_ids].astype('float32'));chunks['eventW'].append(np.full(len(event_ids),window_weight/len(event_ids),dtype='float32'))
   cid=np.flatnonzero(mask&np.tile(event,6))
   if len(cid):chunks['condX'].append(xx[cid]);chunks['condY'].append(yy[cid]);chunks['condW'].append(np.full(len(cid),window_weight/len(cid),dtype='float32'))
 data={k:np.concatenate(v) for k,v in chunks.items()};assert all(np.isfinite(a).all() for a in data.values());meta=dict(panel=panel,condition=condition,windows=windows,features=data['X'].shape[1],rows=len(data['y']),all_eligible_positive_cells=full_positive,all_eligible_transition_cells=full_transition,old_96_cell_expected_positive=old_expected_positive,arrays=list(data),sampling='all queue-positive cells; ongoing additionally all transition cells; <=192 stratified negatives, inverse-inclusion weights',event_negative='only complete six-frame eligible negatives; positives need observed queue')
 for k,v in data.items():np.save(directory/(k+'.npy'),v)
 atomic_json(sentinel,meta);return data,meta
def metric(ss,ps,t):return float(np.mean([window_iou(s['y'],p>=t,s['m']) for s,p in zip(ss,ps) if s['has_future_queue']]))
def raw_predict(s,config,models,xx=None):
 xx=features(s) if xx is None else xx;n=s['x'].shape[1];tag=f"{s['panel']}-{s['condition']}";kind=config['kind'];iteration=config.get('iteration')
 if kind=='direct':return models[tag+'-direct'].predict(xx,num_iteration=iteration,num_threads=4).astype('float32').reshape(6,-1)
 if kind=='event_factor':
  ex=np.concatenate([xx[:n,:185],xx[:n,191:]],axis=1);pe=models[tag+'-event'].predict(ex,num_threads=4);pc=models[tag+'-conditional'].predict(xx,num_threads=4).reshape(6,-1);return (pe[None]*pc).astype('float32')
 if kind=='transition':
  pf=models[tag+'-transition'].predict(xx,num_iteration=iteration,num_threads=4).reshape(6,-1);last,queued=latest_state(s);return np.where(queued[None],1-pf,pf).astype('float32')
 if kind=='speed_residual':
  residual=models[tag+'-speed'].predict(xx,num_threads=4).reshape(6,-1);last,_=latest_state(s);future=last[None]+residual;cut=s['network']['threshold']/s['network']['free_speed'];return (1/(1+np.exp(np.clip((future-cut[None])/config['temperature'],-30,30)))).astype('float32')
 raise ValueError(kind)
def main():
 started=time.time();guard=task1_guard('before');entries=json.loads((R2/'expanded-fit.json').read_text())['samples'];baseline=Baseline(ROOT/'baseline');tune=role_samples('tune');models={};routes={};metadata=[];development=[]
 save('experiment.json',dict(seed=SEED,fit_windows=8192,max_models=48,max_tree_rounds=24000,cpu_threads=4,GPU_training=False,cpu_temperature=None,onset='all observed queue cells + IPW sampled negatives; local precursor features; per-link event probability times conditional horizon queue probability',ongoing='all queue cells plus all observed transitions; direct/transition/speed residual heads',onset_event_rounds=400,onset_conditional_rounds=400,direct_rounds=600,transition_rounds=600,speed_residual_rounds=400,selection='December per-road-condition with half-month stability guard; January global blend/thresholds; February reused report only',Task1='frozen historical55-day S_state=.942828633; hash protection; no offline Task1 predictions in Task2',official_S_queue=None))
 # 在训练之前验收新增特征的未来扰动不变性。
 causal=[]
 for p in PANELS:
  for c in [0,1]:
   e=next(e for e in entries if e['panel']==p and e['condition']==c);s=sample(e);v=copy.deepcopy(arrays(e['day_key']));slot=e['origin_slot']
   for k in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[k][slot+1:]=999
   v['eligible'][slot+1:]=False;b=make_sample(v,s['network'],slot,p);assert np.array_equal(features(s),features(b)) and b['condition']==s['condition'];causal.append(dict(panel=p,condition=c,passed=True))
 save('causality-check.json',dict(passed=True,groups=causal,features_do_not_read_labels=True,Task1_offline_outputs_prohibited=True))
 for p in PANELS:
  for c in [0,1]:
   tag=f'{p}-{c}';progress('building_distribution_data',panel=p,condition=c,completed=len(routes),total=16);data,meta=group_data(entries,p,c);metadata.append(meta);save('sampling-audit.json',metadata);progress('training_group',panel=p,condition=c,rows=meta['rows'],features=meta['features'],all_positive_cells=meta['all_eligible_positive_cells'])
   models[tag+'-direct']=fit(tag+'-direct',data['X'],data['y'],data['w'],600,31 if c==0 else 63,80)
   if c==0:
    models[tag+'-event']=fit(tag+'-event',data['eventX'],data['eventY'],data['eventW'],400,31,60);models[tag+'-conditional']=fit(tag+'-conditional',data['condX'],data['condY'],data['condW'],400,15,40);configs=[dict(kind='direct',iteration=i) for i in [200,400,600]]+[dict(kind='event_factor')]
   else:
    models[tag+'-transition']=fit(tag+'-transition',data['X'],data['flip'],data['w'],600,31,80);models[tag+'-speed']=fit(tag+'-speed',data['X'],data['residual'],data['w'],400,63,80,'huber');configs=[dict(kind=k,iteration=i) for k in ['direct','transition'] for i in [200,400,600]]+[dict(kind='speed_residual',temperature=t) for t in [.025,.05,.1]]
   del data;gc.collect();ss=[s for s in tune if s['panel']==p and s['condition']==c];xx=[features(s) for s in ss];base=[baseline.probability(s) for s in ss];threshold=.15;bv=metric(ss,base,threshold);half=[[i for i,s in enumerate(ss) if (pd.Timestamp(s['origin']).day<=15)==first] for first in [True,False]];candidates=[dict(config=dict(kind='baseline'),alpha_new=0.,proxy=bv,half_gains=[0.,0.],eligible=True)]
   for config in configs:
    raw=[raw_predict(s,config,models,x) for s,x in zip(ss,xx)]
    for alpha in [.25,.5,1.]:
     ps=[alpha*a+(1-alpha)*b for a,b in zip(raw,base)];value=metric(ss,ps,threshold);gains=[]
     for ii in half:gains.append(metric([ss[i] for i in ii],[ps[i] for i in ii],threshold)-metric([ss[i] for i in ii],[base[i] for i in ii],threshold))
     candidates.append(dict(config=config,alpha_new=alpha,proxy=value,half_gains=gains,eligible=value>=bv+.003 and min(gains)>=-.02))
   winner=max((a for a in candidates if a['eligible']),key=lambda a:a['proxy']);routes[(p,c)]=winner;development.append(dict(panel=p,condition=c,baseline_proxy=bv,selected=winner,candidates=candidates));save('december-selection.json',development);save('routes.json',[dict(panel=p,condition=c,**route) for (p,c),route in routes.items()]);del xx;gc.collect()
 def components(ss):
  base=[];adapt=[]
  for s in ss:
   b=baseline.probability(s);r=routes[(s['panel'],s['condition'])];base.append(b);adapt.append(b if r['config']['kind']=='baseline' else r['alpha_new']*raw_predict(s,r['config'],models)+(1-r['alpha_new'])*b)
  return base,adapt
 progress('calibration');cal=role_samples('calibrate');base,adapt=components(cal);candidates=[dict(alpha_distribution=0.,thresholds=[.15,.15],proxy=score(cal,base,[.15,.15])['proxy_IoU'])]
 for alpha in [0.,.25,.5,.75,1.]:
  ps=[alpha*a+(1-alpha)*b for a,b in zip(adapt,base)]
  for t0 in [.05,.075,.1,.15,.2,.25,.35]:
   for t1 in [.075,.1,.15,.2,.25,.35,.5]:candidates.append(dict(alpha_distribution=alpha,thresholds=[t0,t1],proxy=score(cal,ps,[t0,t1])['proxy_IoU']))
 winner=max(candidates,key=lambda a:a['proxy']);hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.txt')};save('frozen-calibration.json',dict(selected=winner,candidates=candidates,hashes=hashes,route_sha256=hashlib.sha256((ROOT/'routes.json').read_bytes()).hexdigest(),selection_month='January',february_status='reused development-after-evaluation',Task1_weights_frozen=True))
 progress('evaluating');audit=role_samples('audit');base,adapt=components(audit);final=[winner['alpha_distribution']*a+(1-winner['alpha_distribution'])*b for a,b in zip(adapt,base)];reports={'official_persistence':score(audit,[official_persistence(s,s['network']) for s in audit]),'carry_T':score(audit,[persistence(s,s['network'],True) for s in audit]),'round4_frozen':score(audit,base,[.15,.15]),'distribution_same_thresholds':score(audit,adapt,[.15,.15]),'selected':score(audit,final,winner['thresholds']),'selected_360min_spacing_diagnostic':score(audit,final,winner['thresholds'],True)};assert abs(reports['round4_frozen']['proxy_IoU']-json.loads((R4/'proxy-results.json').read_text())['reports']['selected']['proxy_IoU'])<1e-12;save('proxy-results.json',dict(official_S_queue=None,reports=reports,selected=winner,new_groups=sum(r['config']['kind']!='baseline' for r in routes.values()),seconds=time.time()-started,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False))
 progress('predicting_official');checks={}
 for split in ['validation','private']:
  frames=[]
  for p in PANELS:
   directory=DATA/'task2'/p/split;index=pd.read_csv(directory/'window_index.csv');template=pd.read_csv(directory/'sample_submission_queue.csv');parts=[]
   for row in index.itertuples():
    s=released_sample(DATA,p,split,row.window_id);bb,aa=components([s]);pr=winner['alpha_distribution']*aa[0]+(1-winner['alpha_distribution'])*bb[0];keys=template[template.window_id==row.window_id].copy();ii={l:i for i,l in enumerate(s['network']['link_ids'])};h=((pd.to_datetime(keys.timestamp,utc=True)-pd.Timestamp(s['origin'])).dt.total_seconds()/300).to_numpy(dtype=int)-1;keys['queue_pred']=(pr[h,keys.link_id.map(ii).to_numpy()]>=winner['thresholds'][s['condition']]).astype(int);parts.append(keys)
   frame=pd.concat(parts,ignore_index=True);checks[split+'/'+p]=validate_queue(frame,template);frame['panel']=p;frames.append(frame)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==87000;frame.to_csv(ROOT/f'queue-{split}.csv',index=False)
 after=task1_guard('after');assert after==guard;assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());save('final-verification.json',dict(passed=True,rows=174000,checks=checks,frozen_hashes_unchanged=True,all_16_causal_feature_checks=True,Task1_92_file_package_unchanged=True,Task1_historical_S_state=.942828633,Task1_score_newly_measured=False,official_S_queue=None));progress('complete',proxy=reports['selected']['proxy_IoU'],new_groups=sum(r['config']['kind']!='baseline' for r in routes.values()),seconds=time.time()-started)
if __name__=='__main__':
 import msvcrt
 lock=open(ROOT/'run.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
 try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 except OSError:raise SystemExit('Already running')
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));progress('failed');raise
