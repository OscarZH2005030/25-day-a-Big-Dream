"""Fixed original-mask February transfer check. Never selects or edits models."""
import common as d
import augmentation_fit as a
import json, time, msvcrt, traceback
import pandas as pd
from datetime import datetime, timezone

O=d.OUT/'augmentation_original_check'
DATES=[f'2031-02-{i:02d}' for i in range(1,29)]

def main():
 while True:
  d.check_deadline()
  state=json.loads((d.WORK/'STATE.json').read_text())
  if state.get('stage')=='augmentation_reserved_complete' and state.get('status')=='stage_complete':break
  if state.get('status')=='failed_or_cutoff':raise RuntimeError('Upstream failed; do not evaluate incomplete policies')
  time.sleep(10)
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b')
 try:
  # Upstream can set completion just before releasing the shared OS lock.
  for attempt in range(30):
   try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
   except OSError:
    if attempt==29:raise
    time.sleep(2)
  policy_path=a.O/'frozen_policy.json';policies=json.loads(policy_path.read_text())['policy']
  initial_hash=d.sha(policy_path);rows=[];daily=[];hashes={}
  for panel in d.night.predictor.load_routes():
   d.check_deadline();d.state(status='running',stage='augmentation_original_february_check',panel=panel,next='Report transfer and failures with policies frozen; no February retuning')
   rp,rm=a.load_reference(panel);sp,sm=a.load_selected(panel,policies[panel]);items=[];outputs={v:[] for v in ['baseline','context','selected']}
   for date in DATES:
    d.check_deadline();item,X=a.original(panel,date);items.append(item);p=item['frame'];values=dict(baseline=p[['speed_kmh','flow_vph']].to_numpy(),context=a.correct(p,X,rm,rp['alpha']),selected=a.correct(p,X,sm,sp['alpha']))
    hashes[panel+'_'+date]=d.sha(d.ROOT/'work/overnight/baseline_predictions'/(panel+'_'+date+'.parquet'))
    for name,v in values.items():
     outputs[name].append(v);daily.append(dict(panel=panel,date=date,variant=name,**d.night.metrics([item],[v])))
   for name,v in outputs.items():rows.append(dict(panel=panel,variant=name,recipe=policies[panel]['recipe'],**d.night.metrics(items,v)))
   pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False);pd.DataFrame(daily).to_csv(O/'by_date.csv',index=False)
   print('ORIGINAL CHECK',panel,rows[-3:],flush=True)
  assert d.sha(policy_path)==initial_hash,'Policy changed during evaluation'
  pd.DataFrame(rows).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'summary.csv')
  model_hashes={str(p.relative_to(d.ROOT)):d.sha(p) for p in (a.O/'models').rglob('*.txt')}
  d.save(O/'integrity.json',dict(policy_sha256=initial_hash,source_sha256=d.sha(__file__),baseline_prediction_sha256=hashes,model_sha256=model_hashes,note='Exposed February development transfer check, no policy retuning. Not official or blind.'))
  d.state(status='stage_complete',stage='augmentation_original_check_complete',next='Assess fixed-policy original-mask and reserved auxiliary results, retain negative results, register next falsifiable study before evaluation.')
 finally:
  try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
  finally:lock.close()

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),dates=DATES,variants=['baseline','context','January-selected augmentation policy'],selection='None; upstream frozen policy only, never change by February results',metrics=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE'],source_sha256=d.sha(__file__),note='Exposed development data; late55 is not evaluated here; Task4 frozen; no Kaggle upload.'))
 try:main()
 except Exception as e:
  d.save(O/'failure.json',dict(error=repr(e),utc=datetime.now(timezone.utc).isoformat()));traceback.print_exc();raise
