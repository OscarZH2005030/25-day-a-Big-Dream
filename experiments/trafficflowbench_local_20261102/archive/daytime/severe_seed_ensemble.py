"""Fixed three-seed ensemble, inherited routes/gamma/tree counts; no seed selection."""
import common as d
import augmentation_fit as a
import severe_augmentation_fit as fit
import numpy as np,pandas as pd,lightgbm as lgb,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'severe_seed_ensemble';PARENT=d.OUT/'multimonth_candidate';SEEDS=[2026110283,2026110284,2026110285]
DATASETS={'mild14':d.WORK/'augmentation_evaluation','mild22':d.WORK/'missingness_selection/february','mild52':d.WORK/'multimonth_robustness','severe82':d.WORK/'severe_augmentation_evaluation'}

def predictions(frame,X,parent,teams,pa,gamma):
 before=a.correct(frame,X,parent,pa)
 if gamma==0:return {n:before.copy() for n in ['parent','single','seed284','seed285','ensemble']}
 corrected=[a.correct(frame,X,mm,1.) for mm in teams];outputs=dict(parent=before,single=before+gamma*(corrected[0]-before),seed284=before+gamma*(corrected[1]-before),seed285=before+gamma*(corrected[2]-before),ensemble=before+gamma*(np.mean(corrected,axis=0)-before));return outputs

def main():
 p=json.loads((fit.O/'protocol.json').read_text());policy=json.loads((fit.O/'frozen_policy.json').read_text())['policy'];records={(r['panel'],r['phase'],r['condition'],r['date']):r for r in pd.read_csv(fit.O/'prepared_cases.csv').to_dict('records')};protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='A fixed average across feature-sampling seeds can reduce prediction variance and sparse-missingness accumulation regressions.',seeds=SEEDS,training_protocol_sha256=d.sha(fit.O/'protocol.json'),training_cases_sha256=d.sha(fit.O/'prepared_cases.csv'),inherited_policy_sha256=d.sha(fit.O/'frozen_policy.json'),source_sha256=d.sha(__file__),selection='None: inherited January-selected gamma/routes and channel tree counts; fixed three-seed arithmetic average of independently clipped augmented predictions before mixing. No best-seed selection, no new early stopping, no February training.',evaluation=dict(original='All February01-28',cached_sets={n:str(root) for n,root in DATASETS.items()},report='All individual seeds and ensemble, all roads, all cached cases, preserve failures'),caveat='Adaptive exposed-development analysis, not independent blind or official Task3/total improvement. Original data/earlier candidates untouched, Task4 fixed, no Kaggle upload.')
 d.save(O/'protocol.json',protocol);models={};parentmodels={}
 for panel,pol in policy.items():
  d.check_deadline();d.state(status='running',stage='severe_ensemble_training',panel=panel,next='Fixed seeds284/285 and inherited tree counts; no seed/road/alpha selection')
  parentmodels[panel]=[lgb.Booster(model_file=str(PARENT/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];first=[lgb.Booster(model_file=str(fit.O/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];models[panel]=[first]
  if pol['gamma']==0:continue
  cases=[('original',*a.original(panel,date)) for date in p['original_training_dates']]
  for condition in p['conditions']:cases.extend((condition,*fit.load_case(panel,'train',condition,date,records)) for date in p['plans'][panel]['train'])
  X,Y,groups=fit.train_arrays(cases)
  for seed in SEEDS[1:]:
   mm=[]
   for channel,name in enumerate(['speed.txt','flow_per_lane.txt']):
    d.check_deadline();path=O/'models'/panel/str(seed)/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():m=lgb.Booster(model_file=str(path))
    else:
     params={k:v for k,v in p['model'].items() if k not in ['max_trees','patience']};params.update(objective='regression',metric='rmse',seed=seed,num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True);m=lgb.train(params,lgb.Dataset(X,label=Y[:,channel],weight=fit.weighted(groups)),num_boost_round=pol['iterations'][channel]);m.save_model(str(path))
    mm.append(m)
   models[panel].append(mm);print('ENSEMBLE FIT',panel,seed,flush=True)
 d.save(O/'frozen_policy.json',dict(policy=policy,seeds=SEEDS,frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=d.sha(O/'protocol.json'),model_sha256={str(path.relative_to(d.ROOT)):d.sha(path) for path in (O/'models').rglob('*.txt')},selection='None; inherited gamma unchanged'))
 rows=[];daily=[]
 for panel,pol in policy.items():
  d.check_deadline();d.state(status='running',stage='severe_ensemble_original_evaluation',panel=panel,next='All fixed seeds and ensemble, exposed February diagnostic; no reselection')
  items=[];values={n:[] for n in ['parent','single','seed284','seed285','ensemble']}
  for i in range(1,29):
   it,X=a.original(panel,f'2031-02-{i:02}');items.append(it);vv=predictions(it['frame'],X,parentmodels[panel],models[panel],pol['parent_alpha'],pol['gamma'])
   for name in values:values[name].append(vv[name]);daily.append(dict(panel=panel,date=it['date'],variant=name,**d.night.metrics([it],[vv[name]])))
  rows.extend(dict(panel=panel,variant=name,**d.night.metrics(items,v)) for name,v in values.items());pd.DataFrame(rows).to_csv(O/'original_by_panel.csv',index=False);pd.DataFrame(daily).to_csv(O/'original_by_date.csv',index=False)
 metrics=['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE'];pd.DataFrame(rows).groupby('variant')[metrics].mean().to_csv(O/'original_summary.csv');rows=[];integrity=[]
 for panel,pol in policy.items():
  for dataset,root in DATASETS.items():
   conditions=['station15','block30'] if dataset=='severe82' else ['point10','block10']
   for condition in conditions:
    d.check_deadline();d.state(status='running',stage='severe_ensemble_cached_evaluation',panel=panel,dataset=dataset,condition=condition,next='Every cached case of all registered sets, keep seed-specific failures; no tuning')
    paths=sorted((root/panel/condition).glob('*/public_train_supervision.parquet'));assert paths;items=[];values={n:[] for n in ['parent','single','seed284','seed285','ensemble']}
    for path in paths:
     folder=path.parent;f=pd.read_parquet(path);z=np.load(folder/'features61.npz');assert str(z['keys_sha256'])==d.sha(folder/'auxiliary_keys.csv');X=z['X'];assert len(f)==len(X);vv=predictions(f,X,parentmodels[panel],models[panel],pol['parent_alpha'],pol['gamma']);items.append(dict(date=folder.name,frame=f))
     for name in values:values[name].append(vv[name])
     integrity.append(dict(dataset=dataset,panel=panel,condition=condition,date=folder.name,feature_sha256=d.sha(folder/'features61.npz'),supervision_sha256=d.sha(path)))
    rows.extend(dict(dataset=dataset,panel=panel,condition=condition,variant=name,dates=len(items),**d.night.metrics(items,v)) for name,v in values.items());pd.DataFrame(rows).to_csv(O/'cached_by_panel.csv',index=False)
  print('ENSEMBLE EVAL',panel,flush=True)
 df=pd.DataFrame(rows);df.groupby(['dataset','condition','variant'])[metrics].mean().to_csv(O/'cached_summary.csv');df.groupby(['condition','variant'])[metrics].mean().to_csv(O/'cached_macro_summary.csv');d.save(O/'cache_provenance.json',integrity);d.state(status='stage_complete',stage='severe_seed_ensemble_complete',next='Assess fixed ensemble versus every individual seed without picking a winner by February; retain all models and negative results, no official claim or uploads.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='severe_seed_ensemble',error=repr(e),next='Repair fixed ensemble execution, preserve all three seeds and inherited gamma/tree counts');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
