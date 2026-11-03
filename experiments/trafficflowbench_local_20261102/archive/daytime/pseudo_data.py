"""Fresh-mask case generation shared by training and sealed evaluation."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import common as d
import fresh_masks as f
sys.path.insert(0,str(d.ROOT/'outputs/overnight/context_candidate'))
from residual_features import build_features
from context_features import extra_features

def generate(panel,date,condition,seed,folder,route,keys,predictors=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);reader=pd.read_parquet;mpath=next((d.DATA/'corridors'/panel/'train/mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));raw=reader(mpath);raw[['timestamp','station_id','link_id']]=raw[['timestamp','station_id','link_id']].astype(str);oldseed=f.SEED
 try:f.SEED=seed;extra=f.mask_selection(raw,panel,date,condition)
 finally:f.SEED=oldseed
 original=keys[keys.timestamp.str[:10]==date].copy();extra['panel']=panel;extra['mask_regime']=original.mask_regime.iloc[0];extra=extra[d.KEYS];assert extra.merge(original,on=d.KEYS).empty;extra.to_csv(folder/'auxiliary_keys.csv',index=False);union=pd.concat([original,extra],ignore_index=True).sort_values(d.KEYS).reset_index(drop=True);assert not union.duplicated(d.KEYS).any();chosen=pd.MultiIndex.from_frame(extra[['timestamp','station_id','link_id']]);hide=pd.MultiIndex.from_frame(raw[['timestamp','station_id','link_id']]).isin(chosen);aug=raw.copy();assert hide.sum()==len(extra)
 for col in ['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm']:
  if col in aug:aug.loc[hide,col]=np.nan
 d.save(folder/'mask_manifest.json',dict(seed=seed,source_sha256=d.sha(mpath),auxiliary_keys_sha256=d.sha(folder/'auxiliary_keys.csv'),original_targets=len(original),auxiliary_targets=len(extra),raw_data_modified=False))
 fd=pd.read_csv(d.DATA/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);length=pd.read_csv(d.DATA/'corridors'/panel/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km
 d.night.predictor.keys_for=lambda data,road,split,which:union.copy()
 def masked_read(path,*args,**kwargs):
  text=str(path).replace('\\','/')
  if '/mainline_states/' in text:raise RuntimeError('Unmasked target truth read during inference')
  if Path(path).resolve()==mpath.resolve():
   cols=kwargs.get('columns',args[0] if args else None);return aug.copy() if cols is None else aug[list(cols)].copy()
  return reader(path,*args,**kwargs)
 pd.read_parquet=masked_read
 try:
  p=d.night.predictor.predict(panel,date,route,d.DATA,folder/'baseline','cuda');p=p[d.KEYS+['speed_kmh','flow_vph']].copy();p['lanes']=p.link_id.map(fd.lanes);X=np.column_stack([build_features(d.DATA,panel,date,p,'train'),extra_features(d.DATA,panel,date,p,'train')]);assert X.shape[1]==61
  union_index=pd.MultiIndex.from_frame(p[d.KEYS]);positions=union_index.get_indexer(pd.MultiIndex.from_frame(extra[d.KEYS]));assert (positions>=0).all();outputs={'baseline':p[['speed_kmh','flow_vph']].to_numpy()[positions]}
  for name,predict in (predictors or {}).items():
   value=predict(p,X);assert np.isfinite(value).all() and (value>=0).all();outputs[name]=value[positions]
 finally:pd.read_parquet=reader
 # No model callback sees the public train supervision below.
 truthpath=next((d.DATA/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));truth=reader(truthpath,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph']);truth[['timestamp','station_id','link_id']]=truth[['timestamp','station_id','link_id']].astype(str);truth=truth.rename(columns={'speed_kmh':'true_v','flow_vph':'true_q'});frame=extra.merge(p,on=d.KEYS,validate='one_to_one').merge(truth,on=['timestamp','station_id','link_id'],validate='one_to_one');frame['length_km']=frame.link_id.map(length);frame['q_lane']=frame.flow_vph/frame.lanes;frame['true_q_lane']=frame.true_q/frame.lanes;assert frame[['true_v','true_q','lanes','length_km']].notna().all().all();frame.to_parquet(folder/'public_train_supervision.parquet',index=False);np.savez_compressed(folder/'features61.npz',X=X[positions],keys_sha256=np.array(d.sha(folder/'auxiliary_keys.csv')));np.savez_compressed(folder/'predictions.npz',**outputs)
 return dict(date=date,frame=frame),X[positions],outputs
