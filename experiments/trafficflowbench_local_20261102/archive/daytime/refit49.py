"""Fixed-recipe refit on Dec14-Jan31; no February supervision or selection."""
import common as d
import augmentation_fit as a
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'refit49';PARENT=d.OUT/'multimonth_candidate'

def original(panel,date,route,keys):
 old=d.ROOT/'work/overnight/baseline_predictions'/(panel+'_'+date+'.parquet')
 if old.exists():return a.original(panel,date)
 folder=d.WORK/'refit49_data'/panel/date;folder.mkdir(parents=True,exist_ok=True);cache=folder/'supervision.parquet';features=folder/'features61.npz'
 if cache.exists() and features.exists():return dict(date=date,frame=pd.read_parquet(cache)),np.load(features)['X']
 reader=pd.read_parquet
 def masked(path,*args,**kw):
  if '/mainline_states/' in str(path).replace('\\','/'):raise RuntimeError('Truth read during inference')
  return reader(path,*args,**kw)
 d.night.predictor.keys_for=lambda data,road,split,which:keys[keys.timestamp.str[:10]==date].copy();pd.read_parquet=masked
 try:
  p=d.night.predictor.predict(panel,date,route,d.DATA,folder/'baseline','cuda')[d.KEYS+['speed_kmh','flow_vph']].copy();fd=pd.read_csv(d.DATA/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);length=pd.read_csv(d.DATA/'corridors'/panel/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km;p['lanes']=p.link_id.map(fd.lanes);X=np.column_stack([a.build_features(d.DATA,panel,date,p,'train'),a.extra_features(d.DATA,panel,date,p,'train')])
 finally:pd.read_parquet=reader
 truthpath=next((d.DATA/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));truth=reader(truthpath,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph']);truth[['timestamp','station_id','link_id']]=truth[['timestamp','station_id','link_id']].astype(str);f=p.merge(truth.rename(columns={'speed_kmh':'true_v','flow_vph':'true_q'}),on=['timestamp','station_id','link_id'],validate='one_to_one');f['length_km']=f.link_id.map(length);f['q_lane']=f.flow_vph/f.lanes;f['true_q_lane']=f.true_q/f.lanes;assert f[['lanes','length_km','true_v','true_q']].notna().all().all();f.to_parquet(cache,index=False);np.savez_compressed(features,X=X);d.save(folder/'provenance.json',dict(date=date,panel=panel,truth_sha256=d.sha(truthpath),keys_sha256=__import__('hashlib').sha256(f[d.KEYS].to_csv(index=False).encode()).hexdigest(),features_sha256=d.sha(features)));return dict(date=date,frame=f),X

def main():
 parent_policy=json.loads((PARENT/'policy.json').read_text())['policy'];selected=json.loads((d.OUT/'multimonth_fit/frozen_policy.json').read_text())['policy'];prior=json.loads((d.OUT/'multimonth_fit/protocol.json').read_text());dates=[f'2030-12-{i:02d}' for i in range(14,32)]+[f'2031-01-{i:02d}' for i in range(1,32)];params=prior['params'];protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),training_dates=dates,evaluation_dates=[f'2031-02-{i:02d}' for i in range(1,29)],recipe='Refit only routes selected for December+January; retain context fallback routes exactly. Fixed selected tree counts/alpha/features/seed, no early stopping or new hyperparameters.',weights='Equal total per month/regime, normalized to training row count',parent_policy_sha256=d.sha(d.OUT/'multimonth_fit/frozen_policy.json'),parent_manifest_sha256=d.sha(PARENT/'manifest.json'),params=params,source_sha256=d.sha(__file__),January_validation_score=None,official_score=None,caveat='January development labels now included in training; no January validation score for refit. February remains residual-model heldout but exposed development, not whole-system blind. Late55 is unrelated and trained February. No Kaggle uploads; Task4 fixed.')
 d.save(O/'protocol.json',protocol);models_by_panel={};provenance={}
 for panel,route in d.night.predictor.load_routes().items():
  d.check_deadline();p=parent_policy[panel];root=O/'models'/panel;root.mkdir(parents=True,exist_ok=True)
  if p['recipe']=='context':models_by_panel[panel]=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];provenance[panel]=dict(recipe='retained_parent',alpha=p['alpha'],features=p['features'],source=str(PARENT/'models'/panel));continue
  assert p['recipe']=='december_january','Unregistered parent recipe'
  d.state(status='running',stage='refit49_training',panel=panel,next='Fixed recipe Dec14-Jan31 refit; no early stopping, February not training, January validation null')
  keys=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);train=[]
  for date in dates:
   d.check_deadline();train.append(original(panel,date,route,keys))
  X=np.concatenate([x for it,x in train]);Y=np.concatenate([(it['frame'][['true_v','true_q_lane']].to_numpy()-it['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for it,x in train]);groups=np.concatenate([np.array([it['date'][:7]+'_'+r for r in it['frame'].mask_regime]) for it,x in train]);models=[]
  for c,n in enumerate(['speed.txt','flow_per_lane.txt']):
   d.check_deadline();iterations=selected[panel]['iterations'][c];m=lgb.train(params,lgb.Dataset(X,label=Y[:,c],weight=a.weights(groups)),num_boost_round=iterations);m.save_model(str(root/n));models.append(m)
  models_by_panel[panel]=models;provenance[panel]=dict(recipe='refit49',alpha=p['alpha'],features=p['features'],iterations=selected[panel]['iterations'],rows=len(X),training_dates=dates,January_validation_score=None,source=str(root));d.save(O/'policy_partial.json',provenance);print('REFIT49',panel,len(X),flush=True)
 d.save(O/'frozen_policy.json',dict(policy=provenance,frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json'),January_validation_score=None,model_sha256={str(p.relative_to(d.ROOT)):d.sha(p) for p in (O/'models').rglob('*.txt')}))
 rows=[];daily=[]
 for panel in parent_policy:
  d.check_deadline();d.state(status='running',stage='refit49_february_evaluation',panel=panel,next='Evaluate fixed refit versus immutable parent, retain all roads; no February selection')
  p=parent_policy[panel];parent=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];new=models_by_panel[panel];items=[];values={n:[] for n in ['parent','refit49']}
  for date in protocol['evaluation_dates']:
   d.check_deadline();it,X=a.original(panel,date);items.append(it)
   for name,mm in [('parent',parent),('refit49',new)]:v=a.correct(it['frame'],X,mm,p['alpha']);values[name].append(v);daily.append(dict(panel=panel,date=date,variant=name,**d.night.metrics([it],[v])))
  rows.extend(dict(panel=panel,variant=n,**d.night.metrics(items,v)) for n,v in values.items());pd.DataFrame(rows).to_csv(O/'february_by_panel.csv',index=False);pd.DataFrame(daily).to_csv(O/'february_by_date.csv',index=False)
 pd.DataFrame(rows).groupby('variant')[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'february_summary.csv');d.state(status='stage_complete',stage='refit49_complete',next='Assess fixed-recipe expanded-label refit with January validation null; fresh auxiliary robustness before any new package, no official claim or submission.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='refit49',error=repr(e),next='Repair execution preserving dates, fixed tree counts and no February training');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
