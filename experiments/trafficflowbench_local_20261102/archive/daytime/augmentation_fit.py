"""Registered balanced control / pseudo-mask augmentation study."""
import json,msvcrt,traceback,sys
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
import common as d
from pseudo_data import generate
sys.path.insert(0,str(d.ROOT/'outputs/overnight/context_candidate'))
from residual_features import build_features
from context_features import extra_features
O=d.OUT/'augmentation'
def original(panel,date):
 pred=pd.read_parquet(d.ROOT/'work/overnight/baseline_predictions'/(panel+'_'+date+'.parquet'))[d.KEYS+['speed_kmh','flow_vph']];fd=pd.read_csv(d.DATA/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);pred['lanes']=pred.link_id.map(fd.lanes);length=pd.read_csv(d.DATA/'corridors'/panel/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km;cache=d.ROOT/'work/overnight/context_features'/panel/(date+'.npz')
 if cache.exists():
  z=np.load(cache);assert str(z['keys'])==__import__('hashlib').sha256(pred[d.KEYS].to_csv(index=False).encode()).hexdigest();X=z['X']
 else:X=np.column_stack([build_features(d.DATA,panel,date,pred,'train'),extra_features(d.DATA,panel,date,pred,'train')])
 truthpath=next((d.DATA/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));truth=pd.read_parquet(truthpath,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph']);truth[['timestamp','station_id','link_id']]=truth[['timestamp','station_id','link_id']].astype(str);f=pred.merge(truth.rename(columns={'speed_kmh':'true_v','flow_vph':'true_q'}),on=['timestamp','station_id','link_id'],validate='one_to_one');f['length_km']=f.link_id.map(length);f['q_lane']=f.flow_vph/f.lanes;f['true_q_lane']=f.true_q/f.lanes
 return dict(date=date,frame=f),X
def weights(groups):
 _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True);return len(groups)/len(counts)/counts[inverse]
def correct(p,X,models,alpha):
 z=np.column_stack([m.predict(X[:,:m.num_feature()],num_threads=4) for m in models]);v=p[['speed_kmh','flow_vph']].to_numpy().copy();v[:,0]+=alpha*z[:,0]*25;v[:,1]+=alpha*z[:,1]*600*p.lanes.to_numpy();return np.maximum(v,0)
def load_reference(panel):
 root=d.ROOT/'outputs/overnight/context_candidate';p=json.loads((root/'policy.json').read_text())['policy'][panel];models=[lgb.Booster(model_file=str(root/'models'/panel/n)) for n in ['speed.txt','flow_per_lane.txt']];return p,models
def load_selected(panel,p):
 if p['recipe']=='reference':return load_reference(panel)
 models=[lgb.Booster(model_file=str(O/'models'/panel/p['recipe']/n)) for n in ['speed.txt','flow_per_lane.txt']];return dict(alpha=p['alpha'],features=61),models
def main():
 protocol=json.loads((O/'protocol.json').read_text());d.save(O/'fit_source.json',dict(source_sha256=d.sha(__file__),protocol_sha256=d.sha(O/'protocol.json')));routes=d.night.predictor.load_routes();policies={};devrows=[]
 for panel in routes:
  d.state(status='running',stage='augmentation_january_fit',panel=panel,next='Fit registered balanced controls, January selection only; February reserve seed must remain sealed until all policies freeze')
  train=[original(panel,date) for date in protocol['original_training_dates']];dev=[original(panel,date) for date in protocol['original_development_dates']];Xo=np.concatenate([a[1] for a in train]);Yo=np.concatenate([(a[0]['frame'][['true_v','true_q_lane']].to_numpy()-a[0]['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for a in train]);go=np.concatenate([np.array(['original_'+v for v in a[0]['frame'].mask_regime]) for a in train]);V=np.concatenate([a[1] for a in dev]);T=np.concatenate([(a[0]['frame'][['true_v','true_q_lane']].to_numpy()-a[0]['frame'][['speed_kmh','q_lane']].to_numpy())/[25,600] for a in dev]);gv=np.concatenate([a[0]['frame'].mask_regime.to_numpy() for a in dev]);items=[a[0] for a in dev];refp,refm=load_reference(panel);ref=d.night.metrics(items,[correct(it['frame'],X,refm,refp['alpha']) for it,X in dev]);best=dict(recipe='reference',alpha=refp['alpha'],metrics=ref)
  xa=[];ya=[];ga=[]
  for condition in protocol['conditions']:
   for date in protocol['plans'][panel]['train']:
    folder=d.WORK/'augmentation_data'/panel/condition/date;f=pd.read_parquet(folder/'public_train_supervision.parquet');xa.append(np.load(folder/'features61.npz')['X']);ya.append((f[['true_v','true_q_lane']].to_numpy()-f[['speed_kmh','q_lane']].to_numpy())/[25,600]);ga.append(np.array([condition+'_'+v for v in f.mask_regime]))
  for recipe in protocol['recipes']:
   if recipe=='balanced_original':X,Y,g=Xo,Yo,go
   else:X=np.concatenate([Xo]+xa);Y=np.concatenate([Yo]+ya);g=np.concatenate([go]+ga)
   models=[]
   for channel,name in enumerate(['speed.txt','flow_per_lane.txt']):
    d.check_deadline();path=O/'models'/panel/recipe/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():m=lgb.Booster(model_file=str(path))
    else:
     params={k:v for k,v in protocol['model'].items() if k not in ['max_trees','patience']};params.update(objective='regression',metric='rmse',num_threads=4,verbosity=-1,deterministic=True,force_col_wise=True);m=lgb.train(params,lgb.Dataset(X,label=Y[:,channel],weight=weights(g)),num_boost_round=protocol['model']['max_trees'],valid_sets=[lgb.Dataset(V,label=T[:,channel],weight=weights(gv))],callbacks=[lgb.early_stopping(protocol['model']['patience'],verbose=False)]);m.save_model(str(path))
    models.append(m)
   for alpha in [.25,.5,1.]:
    score=d.night.metrics(items,[correct(it['frame'],xx,models,alpha) for it,xx in dev]);eligible=score['station_target_deltaN_RMSE']<=ref['station_target_deltaN_RMSE']*1.01;devrows.append(dict(panel=panel,recipe=recipe,alpha=alpha,eligible=eligible,**score))
    if eligible and score['S_state']>best['metrics']['S_state']:best=dict(recipe=recipe,alpha=alpha,iterations=[m.current_iteration() for m in models],metrics=score)
  policies[panel]=best;d.save(O/'policy_partial.json',policies);pd.DataFrame(devrows).to_csv(O/'january_development.csv',index=False);print('AUG FROZEN',panel,best,flush=True)
 d.save(O/'frozen_policy.json',dict(policy=policies,source_sha256=d.sha(__file__),note='Frozen before any evaluation on reserved pseudo-target seed2026110214'));rows=[]
 for panel,route in routes.items():
  d.state(status='running',stage='augmentation_reserved_evaluation',panel=panel,next='Evaluate sealed new-mask seed with fixed policies; do not retune on February')
  record=O/(panel+'_reserved.json')
  if record.exists():rows.extend(json.loads(record.read_text()));continue
  rp,rm=load_reference(panel);sp,sm=load_selected(panel,policies[panel]);keys=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',usecols=d.KEYS,dtype=str)
  for condition in protocol['conditions']:
   items=[];values={v:[] for v in ['baseline','context','selected']}
   for date in protocol['plans'][panel]['evaluation']:
    d.check_deadline();folder=d.WORK/'augmentation_evaluation'/panel/condition/date;callbacks={'context':lambda p,X:correct(p,X,rm,rp['alpha']),'selected':lambda p,X:correct(p,X,sm,sp['alpha'])};item,_,outputs=generate(panel,date,condition,protocol['evaluation_seed'],folder,route,keys,callbacks);items.append(item)
    for name in values:values[name].append(outputs[name])
   rr=[dict(panel=panel,condition=condition,variant=name,recipe=policies[panel]['recipe'],**d.night.metrics(items,pp)) for name,pp in values.items()];rows.extend(rr);print('AUG RESERVED',panel,condition,rr,flush=True)
  d.save(record,[v for v in rows if v['panel']==panel]);pd.DataFrame(rows).to_csv(O/'reserved_by_panel.csv',index=False)
 pd.DataFrame(rows).groupby(['condition','variant'])[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean().to_csv(O/'reserved_summary.csv');d.state(status='stage_complete',stage='augmentation_reserved_complete',next='Assess balanced vs augmented choices and reserved mask transfer; next check original-mask February without retuning; do not claim official/blind performance or upload.')
if __name__=='__main__':
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as err:d.state(status='failed_or_cutoff',stage='augmentation_fit',error=repr(err),next='Fix/resume registered fit without changing dates, weighting or evaluation targets; preserve baselines');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
