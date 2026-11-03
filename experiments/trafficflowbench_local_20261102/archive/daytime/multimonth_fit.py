"""Expand residual training into December; January selection, frozen February check."""
import common as d
import augmentation_fit as a
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'multimonth_fit'

def main():
 prior=json.loads((a.O/'protocol.json').read_text());dates=[f'2030-12-{i:02}' for i in range(14,32)]+prior['original_training_dates'];devdates=prior['original_development_dates'];params={k:v for k,v in prior['model'].items() if k not in ['max_trees','patience']};params.update(objective='regression',metric='rmse',num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True)
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Residual training on one January window limits generalization; adding December with equal month/regime weight improves transfer without February supervision.',train_dates=dates,development_dates=devdates,evaluation_dates=[f'2031-02-{i:02}' for i in range(1,29)],features=61,params=params,rounds=1000,patience=60,control='Reuse balanced_original January models from augmentation; identical features/hyperparameters/seed, only training dates and equal month/regime weighting differ.',weights='Equal total per month and regime; normalized to number of rows',alphas=[.25,.5,1.],selection='Per road maximize January S_state with accumulation-change RMSE <= frozen context reference (no allowed increase). Context fallback. Freeze all roads before February.',source_sha256=d.sha(__file__),input_control_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (a.O/'models').rglob('*.txt') if 'balanced_original' in p.parts},caveat='Exposed public train; December used by published base internal validation, so not a whole-system blind test. No official Task3/total score claim. Task4 baseline fixed, no Kaggle upload.')
 d.save(O/'protocol.json',protocol);policies={};rows=[];hashes={}
 for panel in d.night.predictor.load_routes():
  d.check_deadline();d.state(status='running',stage='multimonth_january_fit',panel=panel,next='Expand training December+January; choose on January only and freeze before February')
  train=[a.original(panel,date) for date in dates];dev=[a.original(panel,date) for date in devdates];X=np.concatenate([xx for it,xx in train]);Y=np.concatenate([(it['frame'][['true_v','true_q_lane']].to_numpy()-it['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for it,xx in train]);group=np.concatenate([np.array([it['date'][:7]+'_'+v for v in it['frame'].mask_regime]) for it,xx in train]);V=np.concatenate([xx for it,xx in dev]);T=np.concatenate([(it['frame'][['true_v','true_q_lane']].to_numpy()-it['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for it,xx in dev]);gv=np.concatenate([it['frame'].mask_regime.to_numpy() for it,xx in dev]);items=[it for it,xx in dev];rp,rm=a.load_reference(panel);reference=d.night.metrics(items,[a.correct(it['frame'],xx,rm,rp['alpha']) for it,xx in dev]);best=dict(recipe='context',alpha=rp['alpha'],metrics=reference);models=[]
  for date in dates+devdates:
   pred=d.ROOT/'work/overnight/baseline_predictions'/(panel+'_'+date+'.parquet');truth=next((d.DATA/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));mask=next((d.DATA/'corridors'/panel/'train/mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'))
   for kind,p in [('baseline',pred),('truth',truth),('masked',mask)]:hashes[panel+'/'+date+'/'+kind]=d.sha(p)
  for channel,name in enumerate(['speed.txt','flow_per_lane.txt']):
   d.check_deadline();m=lgb.train(params,lgb.Dataset(X,label=Y[:,channel],weight=a.weights(group)),num_boost_round=1000,valid_sets=[lgb.Dataset(V,label=T[:,channel],weight=a.weights(gv))],callbacks=[lgb.early_stopping(60,verbose=False)]);path=O/'models'/panel/name;path.parent.mkdir(parents=True,exist_ok=True);m.save_model(str(path));models.append(m)
  control=[lgb.Booster(model_file=str(a.O/'models'/panel/'balanced_original'/name)) for name in ['speed.txt','flow_per_lane.txt']]
  rows.append(dict(panel=panel,recipe='context',alpha=rp['alpha'],eligible=True,**reference))
  for recipe,mm in [('january_control',control),('december_january',models)]:
   for alpha in protocol['alphas']:
    score=d.night.metrics(items,[a.correct(it['frame'],xx,mm,alpha) for it,xx in dev]);eligible=score['station_target_deltaN_RMSE']<=reference['station_target_deltaN_RMSE'];rows.append(dict(panel=panel,recipe=recipe,alpha=alpha,eligible=eligible,**score))
    if eligible and score['S_state']>best['metrics']['S_state']:best=dict(recipe=recipe,alpha=alpha,iterations=[m.current_iteration() for m in mm],metrics=score)
  policies[panel]=best;pd.DataFrame(rows).to_csv(O/'january_by_panel.csv',index=False);d.save(O/'policy_partial.json',policies);print('MULTIMONTH FROZEN',panel,best,flush=True)
 d.save(O/'training_data_sha256.json',hashes);d.save(O/'frozen_policy.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),policy=policies,protocol_sha256=d.sha(O/'protocol.json'),model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (O/'models').rglob('*.txt')}))
 if all(p['recipe']=='context' for p in policies.values()):
  d.save(O/'evaluation_skipped.json',dict(reason='January chose only unchanged context; skip duplicate February evaluation',new_score=None));d.state(status='stage_complete',stage='multimonth_fit_complete',next='Retain negative multi-month result; no unselected February screening.');return
 rows=[];daily=[]
 for panel in d.night.predictor.load_routes():
  d.check_deadline();d.state(status='running',stage='multimonth_february_evaluation',panel=panel,next='Frozen January policy development transfer only; no retuning')
  rp,rm=a.load_reference(panel);p=policies[panel]
  if p['recipe']=='context':sm=rm
  else:
   root=O/'models'/panel if p['recipe']=='december_january' else a.O/'models'/panel/'balanced_original';sm=[lgb.Booster(model_file=str(root/name)) for name in ['speed.txt','flow_per_lane.txt']]
  items=[];values={name:[] for name in ['baseline','context','selected']}
  for date in protocol['evaluation_dates']:
   d.check_deadline();it,xx=a.original(panel,date);items.append(it);f=it['frame'];vv=dict(baseline=f[['speed_kmh','flow_vph']].to_numpy(),context=a.correct(f,xx,rm,rp['alpha']),selected=a.correct(f,xx,sm,p['alpha']))
   for name,val in vv.items():values[name].append(val);daily.append(dict(panel=panel,date=date,variant=name,**d.night.metrics([it],[val])))
  rows.extend(dict(panel=panel,variant=name,recipe=p['recipe'],**d.night.metrics(items,vv)) for name,vv in values.items());pd.DataFrame(rows).to_csv(O/'february_by_panel.csv',index=False);pd.DataFrame(daily).to_csv(O/'february_by_date.csv',index=False)
 pd.DataFrame(rows).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_summary.csv');d.state(status='stage_complete',stage='multimonth_fit_complete',next='Inspect fixed original-mask transfer and failures; test fresh auxiliary masks before packaging any new candidate. No official claim or upload.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='multimonth_fit',error=repr(e),next='Repair execution without changing registered training, January selection or evaluation dates');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
