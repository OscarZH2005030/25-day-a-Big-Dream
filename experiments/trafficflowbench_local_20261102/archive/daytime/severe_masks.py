"""Frozen candidate stress: all-day station outages and longer observation gaps."""
import common as d
import augmentation_fit as a
import pseudo_data as g
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback,hashlib
from datetime import datetime,timezone
O=d.OUT/'severe_masks';PARENT=d.OUT/'multimonth_candidate';REFIT=d.OUT/'refit49';SEED=2026110272

def selector(raw,panel,date,condition):
 valid=raw.is_score_eligible.astype(bool)&(raw.pct_observed>=75)&raw.speed_kmh.notna()&raw.flow_vph.notna()&raw.occupancy.notna();available=raw.loc[valid,['timestamp','station_id','link_id']].copy().astype(str).sort_values(['station_id','timestamp']).reset_index(drop=True);seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}/{date}/{condition}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
 if condition=='station15':
  stations=sorted(available.station_id.unique());chosen=rng.choice(stations,max(1,int(np.ceil(.15*len(stations)))),replace=False);selected=available[available.station_id.isin(chosen)]
 elif condition=='block30':
  pieces=[]
  for station,p in available.groupby('station_id',sort=True):
   ts=pd.to_datetime(p.timestamp,utc=True);ticks=(ts.dt.hour*12+ts.dt.minute//5).to_numpy();chosen=np.zeros(len(p),bool);budget=max(1,int(.30*len(p)))
   for start in rng.permutation(265):
    chosen|=(ticks>=start)&(ticks<start+24)
    if chosen.sum()>=budget:break
   pieces.append(p.loc[chosen])
  selected=pd.concat(pieces,ignore_index=True)
 else:raise ValueError(condition)
 return selected.sort_values(['timestamp','station_id','link_id']).reset_index(drop=True)

def main():
 routes=d.night.predictor.load_routes();pp=json.loads((PARENT/'policy.json').read_text())['policy'];refit=json.loads((REFIT/'frozen_policy.json').read_text())['policy'];plans={};templates={}
 for panel in routes:
  k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);templates[panel]=k;v=k.assign(date=k.timestamp.str[:10]);v=v[(v.date>='2031-02-03')&(v.date<='2031-02-27')];plans[panel]=sorted(v.groupby('mask_regime').date.apply(lambda z:sorted(set(z))[len(set(z))//2]).tolist())
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Expanded-label residual calibration may fail under station outages or longer gaps despite original-mask state improvements.',seed=SEED,plans=plans,conditions=dict(station15='Random ceil15% of stations, hide every eligible currently observed cell all day',block30='Hide about30% of currently observed eligible cells per station using randomly located two-hour blocks'),variants=['baseline','context','multimonth','refit49'],selection='None; all policies frozen, all roads retained, no per-road tuning after this evaluation',source_sha256=d.sha(__file__),generator_sha256=d.sha(d.WORK/'pseudo_data.py'),parent_policy_sha256=d.sha(PARENT/'policy.json'),refit_policy_sha256=d.sha(REFIT/'frozen_policy.json'),scope='Artificial sensitivity on exposed public train, no official Task3/LWR or whole-system blind claim. Task4 baseline frozen, no uploads.')
 d.save(O/'protocol.json',protocol)
 # Complete provenance of refit supervision, including newly inferred Jan01-04.
 hashes={};training_dates=json.loads((REFIT/'protocol.json').read_text())['training_dates']
 for panel,p in refit.items():
  if p['recipe']!='refit49':continue
  for date in training_dates:
   for kind in ['mainline_states','mainline_states_masked']:
    path=next((d.DATA/'corridors'/panel/'train'/kind).rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));hashes[panel+'/'+date+'/'+kind]=d.sha(path)
  for name in ['links.csv','fd_parameters.csv']:
   path=d.DATA/'corridors'/panel/'network'/name;hashes[panel+'/network/'+name]=d.sha(path)
 d.save(REFIT/'training_data_sha256.json',dict(files=hashes,source_protocol_sha256=d.sha(REFIT/'protocol.json'),scope='Only refitted routes; retained parent routes retain parent provenance.'))
 old_selector=g.f.mask_selection;g.f.mask_selection=selector;rows=[]
 try:
  for panel,route in routes.items():
   rp,rm=a.load_reference(panel);parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];root=REFIT/'models'/panel if refit[panel]['recipe']=='refit49' else PARENT/'models'/panel;new=[lgb.Booster(model_file=str(root/n)) for n in ['speed.txt','flow_per_lane.txt']]
   for condition in protocol['conditions']:
    items=[];values={name:[] for name in protocol['variants']}
    for date in plans[panel]:
     d.check_deadline();d.state(status='running',stage='severe_missingness_evaluation',panel=panel,condition=condition,date=date,next='Frozen station/long-gap stress; retain all tradeoffs without selecting by results')
     callbacks=dict(context=lambda f,X:a.correct(f,X,rm,rp['alpha']),multimonth=lambda f,X:a.correct(f,X,parent,pp[panel]['alpha']),refit49=lambda f,X:a.correct(f,X,new,pp[panel]['alpha']));item,_,outputs=g.generate(panel,date,condition,SEED,d.WORK/'severe_masks'/panel/condition/date,route,templates[panel],callbacks);items.append(item)
     for name in values:values[name].append(outputs[name])
    rows.extend(dict(panel=panel,condition=condition,variant=name,**d.night.metrics(items,vv)) for name,vv in values.items());pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False);print('SEVERE',panel,condition,flush=True)
 finally:g.f.mask_selection=old_selector
 metrics=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE'];df=pd.DataFrame(rows);df.groupby(['condition','variant'])[metrics].mean().to_csv(O/'summary.csv');rng=np.random.default_rng(2026110273);boot=[]
 for condition,gg in df.groupby('condition'):
  ref=gg[gg.variant=='context'].set_index('panel')
  for name in ['multimonth','refit49']:
   vv=gg[gg.variant==name].set_index('panel').loc[ref.index];delta=vv[metrics].to_numpy()-ref[metrics].to_numpy();reps=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
   for i,metric in enumerate(metrics):boot.append(dict(condition=condition,variant=name,metric=metric,difference=float(delta[:,i].mean()),conditional_low=float(np.quantile(reps[:,i],.025)),conditional_high=float(np.quantile(reps[:,i],.975))))
 d.save(O/'conditional_road_bootstrap.json',dict(seed=2026110273,replicates=2000,scope='Paired resampling of ten known roads; not official or population uncertainty',results=boot));d.state(status='stage_complete',stage='severe_masks_complete',next='Assess frozen station-outage and long-gap results and failures; preserve parent/refit separately, no official claims or uploads.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='severe_masks',error=repr(e),next='Repair frozen stress execution without changing registered masks or models');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
