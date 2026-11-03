"""Register and generate fresh January pseudo-mask training, reserving evaluation."""
import json,msvcrt,traceback
import numpy as np,pandas as pd
import common as d
from pseudo_data import generate
O=d.OUT/'augmentation';O.mkdir(parents=True,exist_ok=True)
def main():
 routes=d.night.predictor.load_routes();plans={};templates={}
 for panel in routes:
  k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str);templates[panel]=k;v=k.assign(date=k.timestamp.str[:10]);jan=v[(v.date>='2031-01-05')&(v.date<='2031-01-20')];feb=v[(v.date>='2031-02-08')&(v.date<='2031-02-22')];plans[panel]=dict(train=sorted(set(jan.groupby('mask_regime').date.min())|set(jan.groupby('mask_regime').date.max())),evaluation=sorted(feb.groupby('mask_regime').date.apply(lambda z:sorted(set(z))[len(set(z))//2]).tolist()))
 protocol=dict(scope='January pseudo-mask augmentation; February is exposed source data and not whole-system blind. Reserve new seed/cells for fixed evaluation.',plans=plans,train_seed=2026110212,evaluation_seed=2026110214,conditions=['point10','block10'],original_training_dates=[f'2031-01-{i:02}' for i in range(5,21)],original_development_dates=[f'2031-01-{i:02}' for i in range(21,32)],recipes=['balanced_original','augmented_balanced'],training_weights='Equal total per (source_kind,regime), normalized to number of rows; original-only control equal per regime.',model=dict(num_leaves=31,min_data_in_leaf=1000,learning_rate=.03,lambda_l2=20.,feature_fraction=.9,max_trees=1000,patience=60,seed=2026110213),selection='January original dev S_state maximum with accumulation RMSE <=1.01*context reference; fixed context fallback; alpha .25/.5/1',feature_source_sha256=d.sha(d.ROOT/'work/overnight/context_features.py'),generator_sha256=d.sha(__import__('pathlib').Path(__file__).with_name('pseudo_data.py')),source_sha256=d.sha(__file__),no_original_data_changes=True,no_kaggle_upload=True)
 if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol
 else:d.save(O/'protocol.json',protocol)
 rows=[]
 for panel,route in routes.items():
  for condition in protocol['conditions']:
   for date in plans[panel]['train']:
    d.check_deadline();d.state(status='running',stage='prepare_pseudo_augmentation',panel=panel,condition=condition,date=date,next='Finish January pseudo-mask data, then fit registered balanced controls. Do not generate reserved February evaluation before policy freeze.')
    folder=d.WORK/'augmentation_data'/panel/condition/date
    if (folder/'public_train_supervision.parquet').exists() and (folder/'features61.npz').exists():f=pd.read_parquet(folder/'public_train_supervision.parquet')
    else:item,_,_=generate(panel,date,condition,protocol['train_seed'],folder,route,templates[panel]);f=item['frame']
    rows.append(dict(panel=panel,condition=condition,date=date,rows=len(f),regime=f.mask_regime.iloc[0]));pd.DataFrame(rows).to_csv(O/'training_cases.csv',index=False);print('aug prepared',panel,condition,date,len(f),flush=True)
 d.state(status='stage_complete',stage='pseudo_augmentation_ready',next='Fit balanced-original and augmented-balanced controls according to outputs/daytime/augmentation/protocol.json, January selection only. Reserved seed2026110214 remains unseen.')
if __name__=='__main__':
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as err:d.state(status='failed_or_cutoff',stage='prepare_pseudo_augmentation',error=repr(err),next='Repair generator or schema without changing registered dates, masks, seeds or evaluation rule');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
