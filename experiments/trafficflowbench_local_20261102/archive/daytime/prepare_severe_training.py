"""Registered December/January severe-mask augmentation data preparation."""
import common as d
import pseudo_data as g
import severe_masks as stress
import numpy as np,pandas as pd,json,msvcrt,traceback
from datetime import datetime,timezone
O=d.OUT/'severe_augmentation'

def main():
 routes=d.night.predictor.load_routes();templates={};plans={}
 for panel in routes:
  k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);templates[panel]=k;v=k.assign(date=k.timestamp.str[:10]);plans[panel]={};train=[]
  for lo,hi in [('2030-12-14','2030-12-31'),('2031-01-05','2031-01-20')]:
   z=v[(v.date>=lo)&(v.date<=hi)];train.extend(set(z.groupby('mask_regime').date.min())|set(z.groupby('mask_regime').date.max()))
  plans[panel]['train']=sorted(train)
  for name,lo,hi in [('development','2031-01-21','2031-01-31'),('evaluation','2031-02-03','2031-02-27')]:
   z=v[(v.date>=lo)&(v.date<=hi)];plans[panel][name]=sorted(z.groupby('mask_regime').date.apply(lambda x:sorted(set(x))[len(set(x))//2]).tolist())
 protocol=dict(registered_utc=datetime.now(timezone.utc).isoformat(),hypothesis='Train on matched severe missingness in December/January to reduce distribution shift while retaining original-mask quality.',plans=plans,conditions=['station15','block30'],training_seed=2026110280,development_seed=2026110281,evaluation_seed=2026110282,model_seed=2026110283,original_training_dates=[f'2030-12-{i:02}' for i in range(14,32)]+[f'2031-01-{i:02}' for i in range(5,21)],original_development_dates=[f'2031-01-{i:02}' for i in range(21,32)],features=61,model=dict(num_leaves=31,min_data_in_leaf=1000,learning_rate=.03,lambda_l2=20,feature_fraction=.9,max_trees=1000,patience=60),weights='50% original,25% station15,25% block30; within source equal total per month/regime for training, per regime for development; normalized to row count.',selection='Per road select gamma .25/.5/1 blending augmented model output with frozen multimonth parent. Maximize equal mean January S_state over original/station15/block30, subject original state >= parent-0.0001 and every condition accumulation-change RMSE <=parent*1.01. Gamma0 fallback. Freeze all roads before new February seed.',selector_sha256=d.sha(d.WORK/'severe_masks.py'),generator_sha256=d.sha(d.WORK/'pseudo_data.py'),source_sha256=d.sha(__file__),parent_manifest_sha256=d.sha(d.OUT/'multimonth_candidate/manifest.json'),scope='Adaptively reused public development data, no whole-system blind or official Task3 claim. February masks remain ungenerated until all January choices frozen. Task4 baseline fixed. No Kaggle upload.')
 path=O/'protocol.json'
 if path.exists():
  old=json.loads(path.read_text());protocol['registered_utc']=old['registered_utc'];assert old==protocol
 else:d.save(path,protocol)
 rows=[];oldselector=g.f.mask_selection;oldseed=stress.SEED;g.f.mask_selection=stress.selector
 try:
  for panel,route in routes.items():
   for phase,seed in [('train',protocol['training_seed']),('development',protocol['development_seed'])]:
    stress.SEED=seed
    for condition in protocol['conditions']:
     for date in plans[panel][phase]:
      d.check_deadline();d.state(status='running',stage='severe_augmentation_prepare',panel=panel,phase=phase,condition=condition,date=date,next='Complete registered December/January severe-mask train/dev data; February reserve untouched, single shared lock')
      folder=d.WORK/'severe_augmentation_data'/phase/panel/condition/date;manifest=folder/'mask_manifest.json';cache=folder/'public_train_supervision.parquet';features=folder/'features61.npz'
      if manifest.exists() and cache.exists() and features.exists():
       m=json.loads(manifest.read_text());assert m['seed']==seed and m['auxiliary_keys_sha256']==d.sha(folder/'auxiliary_keys.csv');f=pd.read_parquet(cache)
      else:it,_,_=g.generate(panel,date,condition,seed,folder,route,templates[panel]);f=it['frame']
      rows.append(dict(panel=panel,phase=phase,condition=condition,date=date,rows=len(f),regime=f.mask_regime.iloc[0],mask_manifest_sha256=d.sha(manifest),features_sha256=d.sha(features),supervision_sha256=d.sha(cache)));pd.DataFrame(rows).to_csv(O/'prepared_cases.csv',index=False);print('SEVERE PREP',phase,panel,condition,date,len(f),flush=True)
 finally:g.f.mask_selection=oldselector;stress.SEED=oldseed
 d.state(status='stage_complete',stage='severe_augmentation_data_ready',next='Train registered matched-severity models and blend selection only on January. Frozen parent remains unchanged; February seed2026110282 not generated yet.')

if __name__=='__main__':
 O.mkdir(parents=True,exist_ok=True);lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as e:d.state(status='failed_or_cutoff',stage='prepare_severe_training',error=repr(e),next='Resume registered data generation with same dates/seeds; do not change evaluation or source models');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
