"""Frozen mixture on ALL three previously generated mild-mask seed sets."""
import common as d
import augmentation_fit as a
import severe_augmentation_fit as fit
import numpy as np,pandas as pd,lightgbm as lgb,json,time,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'mild_cached_transfer';PARENT=d.OUT/'multimonth_candidate';PACKAGE=d.OUT/'severe_mixture_candidate'
SETS={'seed2026110214':d.WORK/'augmentation_evaluation','seed2026110222':d.WORK/'missingness_selection/february','seed2026110252':d.WORK/'multimonth_robustness'}
SOURCE_PROTOCOLS={'seed2026110214':d.OUT/'augmentation/protocol.json','seed2026110222':d.OUT/'missingness_selection/protocol.json','seed2026110252':d.OUT/'multimonth_robustness/protocol.json'}

def main():
 while True:
  d.check_deadline();state=json.loads((d.WORK/'STATE.json').read_text())
  acceptance=d.OUT/'scored_severe_mixture_parts/acceptance.json'
  if acceptance.exists():
   verified=json.loads(acceptance.read_text())
   if verified['total_rows']==6735795 and len(verified['files'])==20 and verified['package_manifest_sha256']==d.sha(PACKAGE/'manifest.json'):break
  if state.get('stage')=='severe_mixture_package_complete' and state.get('status')=='stage_complete':break
  if state.get('status')=='failed_or_cutoff':raise RuntimeError('Package requires repair before transfer audit')
  time.sleep(10)
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');acquired=False
 try:
  for attempt in range(30):
   try:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);acquired=True;break
   except OSError:
    if attempt==29:raise
    time.sleep(2)
  policies=json.loads((fit.O/'frozen_policy.json').read_text())['policy'];parentpolicy=json.loads((PARENT/'policy.json').read_text())['policy'];refit=json.loads((d.OUT/'refit49/frozen_policy.json').read_text())['policy'];frozenhash=d.sha(fit.O/'frozen_policy.json');d.save(O/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),source_sets={k:str(p) for k,p in SETS.items()},conditions=['point10','block10'],selection='None; evaluate every available case of all three registered prior mild-mask sets. Do not choose roads/seeds after results.',variants=['baseline','context','multimonth','refit49','severe_mixture'],cached_features='Computed under the actual auxiliary mask before truth read; verify keys and reproduce original context predictions before evaluating new model.',policy_sha256=frozenhash,package_manifest_sha256=d.sha(PACKAGE/'manifest.json'),source_sha256=d.sha(__file__),scope='Adaptively reused exposed development observations, not blind/official. Sources may overlap, not independent tests. No uploads.'))
  sourceplans={k:json.loads(p.read_text()) for k,p in SOURCE_PROTOCOLS.items()};d.save(O/'source_protocol_integrity.json',dict(files={k:dict(path=str(p),sha256=d.sha(p)) for k,p in SOURCE_PROTOCOLS.items()},repair='Replace false three-date assumption with exact original registered date sets. No masks, models or selection changed.'))
  rows=[];hashes=[]
  for panel in parentpolicy:
   rp,rm=a.load_reference(panel);parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];new=[lgb.Booster(model_file=str(fit.O/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];root=d.OUT/'refit49/models'/panel if refit[panel]['recipe']=='refit49' else PARENT/'models'/panel;refitmodels=[lgb.Booster(model_file=str(root/n)) for n in ['speed.txt','flow_per_lane.txt']];p=policies[panel];pa=parentpolicy[panel]['alpha']
   for dataset,root in SETS.items():
    for condition in ['point10','block10']:
     d.check_deadline();d.state(status='running',stage='mild_cached_transfer',panel=panel,dataset=dataset,condition=condition,next='All prior mild-mask sets, no retuning; cached masked features must reproduce old context output')
     cases=sorted((root/panel/condition).glob('*/public_train_supervision.parquet'))
     if dataset=='seed2026110214':expected=sourceplans[dataset]['plans'][panel]['evaluation']
     elif dataset=='seed2026110222':expected=sourceplans[dataset]['plans'][panel]['february']
     else:expected=sourceplans[dataset]['plans'][panel]
     assert sorted(p.parent.name for p in cases)==sorted(expected),(dataset,panel,condition,'registered dates differ');items=[];values={v:[] for v in ['baseline','context','multimonth','refit49','severe_mixture']}
     for path in cases:
      folder=path.parent;f=pd.read_parquet(path);keys=pd.read_csv(folder/'auxiliary_keys.csv',dtype=str);assert f[d.KEYS].reset_index(drop=True).equals(keys[d.KEYS]);z=np.load(folder/'features61.npz');assert str(z['keys_sha256'])==d.sha(folder/'auxiliary_keys.csv');X=z['X'];assert len(X)==len(f);pred=np.load(folder/'predictions.npz');baseline=f[['speed_kmh','flow_vph']].to_numpy();context=a.correct(f,X,rm,rp['alpha']);assert np.allclose(context,pred['context'],atol=1e-9,rtol=0) and np.allclose(baseline,pred['baseline'],atol=1e-10,rtol=0);items.append(dict(date=folder.name,frame=f));vv=dict(baseline=baseline,context=context,multimonth=a.correct(f,X,parent,pa),refit49=a.correct(f,X,refitmodels,pa),severe_mixture=fit.mixture(f,X,parent,new,p['parent_alpha'],p['gamma']))
      for name in values:values[name].append(vv[name])
      hashes.append(dict(dataset=dataset,panel=panel,condition=condition,date=folder.name,features_sha256=d.sha(folder/'features61.npz'),supervision_sha256=d.sha(path),keys_sha256=d.sha(folder/'auxiliary_keys.csv'),original_context_reproduced=True))
     regimes=sorted(set().union(*(set(it['frame'].mask_regime) for it in items)))
     rows.extend(dict(dataset=dataset,panel=panel,condition=condition,variant=name,dates=len(items),available_regimes='|'.join(regimes),**d.night.metrics(items,v)) for name,v in values.items());pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False)
   print('MILD TRANSFER',panel,flush=True)
  assert d.sha(fit.O/'frozen_policy.json')==frozenhash
  df=pd.DataFrame(rows);metrics=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE'];df.groupby(['dataset','condition','variant'])[metrics].mean().to_csv(O/'summary_by_seed.csv');df.groupby(['condition','variant'])[metrics].mean().to_csv(O/'macro_across_existing_seeds.csv');d.save(O/'cache_integrity.json',hashes);d.state(status='stage_complete',stage='mild_cached_transfer_complete',next='Assess all mild seeds without tuning; preserve package and all failures. Investigate new hypothesis only with preregistered January choices, not February road selection.')
 finally:
  if acquired:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
  lock.close()

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True)
 try:main()
 except Exception as e:d.save(O/'failure.json',dict(error=repr(e),utc=datetime.now(timezone.utc).isoformat()));traceback.print_exc();raise
