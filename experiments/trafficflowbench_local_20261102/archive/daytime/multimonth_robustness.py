"""Fixed-policy auxiliary masks and conditional road bootstrap; no tuning."""
import common as d
import augmentation_fit as a
from pseudo_data import generate
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,time,traceback
from datetime import datetime,timezone
O=d.OUT/'multimonth_robustness';SOURCE=d.OUT/'multimonth_fit'

def main():
 while True:
  d.check_deadline();state=json.loads((d.WORK/'STATE.json').read_text())
  if state.get('stage')=='multimonth_fit_complete' and state.get('status')=='stage_complete':break
  if state.get('status')=='failed_or_cutoff':raise RuntimeError('Upstream fit requires repair')
  time.sleep(10)
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');acquired=False
 try:
  for attempt in range(30):
   try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);acquired=True;break
   except OSError:
    if attempt==29:raise
    time.sleep(2)
  path=SOURCE/'frozen_policy.json';policy_hash=d.sha(path);policies=json.loads(path.read_text())['policy'];plans={};templates={};routes=d.night.predictor.load_routes()
  for panel in routes:
   k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);templates[panel]=k;v=k.assign(date=k.timestamp.str[:10]);v=v[(v.date>='2031-02-03')&(v.date<='2031-02-27')];plans[panel]=sorted(v.groupby('mask_regime').date.apply(lambda z:sorted(set(z))[len(set(z))//2]).tolist())
  d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),plans=plans,seed=2026110252,conditions=['point10','block10'],policy_sha256=policy_hash,source_sha256=d.sha(__file__),generator_sha256=d.sha(d.WORK/'pseudo_data.py'),selection='None: use already frozen January policy; retain every road, no February-based fallback',bootstrap=dict(seed=2026110253,replicates=2000,unit='whole road, paired across variants; conditional on available ten roads and development dates'),caveat='Fresh masks on exposed public train, not whole-system blind, not official Task3/leaderboard. Task4 fixed; no uploads.'))
  rows=[]
  for panel,route in routes.items():
   rp,rm=a.load_reference(panel);p=policies[panel]
   if p['recipe']=='context':sm=rm
   else:
    root=SOURCE/'models'/panel if p['recipe']=='december_january' else a.O/'models'/panel/'balanced_original';sm=[lgb.Booster(model_file=str(root/n)) for n in ['speed.txt','flow_per_lane.txt']]
   for condition in ['point10','block10']:
    items=[];values={n:[] for n in ['baseline','context','selected']}
    for date in plans[panel]:
     d.check_deadline();d.state(status='running',stage='multimonth_auxiliary_evaluation',panel=panel,condition=condition,date=date,next='Frozen January policy on new pseudo-mask seed, retain every failure without retuning')
     callbacks=dict(context=lambda f,X:a.correct(f,X,rm,rp['alpha']),selected=lambda f,X:a.correct(f,X,sm,p['alpha']));it,_,vv=generate(panel,date,condition,2026110252,d.WORK/'multimonth_robustness'/panel/condition/date,route,templates[panel],callbacks);items.append(it)
     for n in values:values[n].append(vv[n])
    rows.extend(dict(panel=panel,condition=condition,variant=n,recipe=p['recipe'],**d.night.metrics(items,vv)) for n,vv in values.items());pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False);print('MULTIMONTH ROBUST',panel,condition,flush=True)
  assert d.sha(path)==policy_hash,'Frozen policy changed'
  df=pd.DataFrame(rows);metrics=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE'];df.groupby(['condition','variant'])[metrics].mean().to_csv(O/'summary.csv');rng=np.random.default_rng(2026110253);intervals=[]
  for cond,g in df.groupby('condition'):
   ref=g[g.variant=='context'].set_index('panel');sel=g[g.variant=='selected'].set_index('panel').loc[ref.index];delta=sel[metrics].to_numpy()-ref[metrics].to_numpy();reps=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
   for i,metric in enumerate(metrics):intervals.append(dict(condition=cond,metric=metric,selected_minus_context=float(delta[:,i].mean()),conditional_road_bootstrap_low=float(np.quantile(reps[:,i],.025)),conditional_road_bootstrap_high=float(np.quantile(reps[:,i],.975)),roads_improved=int((delta[:,i]>0).sum() if metric=='S_state' else (delta[:,i]<0).sum()),roads_worsened=int((delta[:,i]<0).sum() if metric=='S_state' else (delta[:,i]>0).sum())))
  d.save(O/'paired_bootstrap.json',dict(scope='Conditional resampling of ten known roads, not population/official uncertainty',results=intervals));d.state(status='stage_complete',stage='multimonth_robustness_complete',next='Assess frozen state/accumulation improvements and per-road failures, package only with caveats; no Kaggle uploads or model selection from these results.')
 finally:
  if acquired:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
  lock.close()

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 try:main()
 except Exception as e:d.save(O/'failure.json',dict(error=repr(e),utc=datetime.now(timezone.utc).isoformat()));traceback.print_exc();raise
