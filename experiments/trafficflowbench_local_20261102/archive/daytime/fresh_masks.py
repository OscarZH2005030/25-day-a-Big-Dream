"""Fresh auxiliary-target robustness: all masking occurs in RAM."""
import json,hashlib,importlib.util,sys,msvcrt,traceback
from pathlib import Path
import numpy as np,pandas as pd
import common as d
e=d.night;e.torch.set_num_threads(4)
O=d.OUT/'fresh_masks';O.mkdir(parents=True,exist_ok=True)
SEED=20261102;CONDITIONS=['point10','block10']
def mask_selection(raw,panel,date,condition):
 valid=raw.is_score_eligible.astype(bool)&(raw.pct_observed>=75)&raw.speed_kmh.notna()&raw.flow_vph.notna()&raw.occupancy.notna();available=raw.loc[valid,['timestamp','station_id','link_id']].copy().astype(str).sort_values(['station_id','timestamp']).reset_index(drop=True)
 seed=int.from_bytes(hashlib.sha256(f'{SEED}/{panel}/{date}/{condition}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
 if condition=='point10':selected=available.iloc[np.sort(rng.choice(len(available),max(1,int(.1*len(available))),replace=False))].copy()
 else:
  pieces=[]
  for station,g in available.groupby('station_id',sort=True):
   ts=pd.to_datetime(g.timestamp,utc=True);ticks=(ts.dt.hour*12+ts.dt.minute//5).to_numpy();chosen=np.zeros(len(g),bool);budget=max(1,int(.1*len(g)));starts=rng.permutation(283)
   for start in starts:
    chosen|=(ticks>=start)&(ticks<start+6)
    if chosen.sum()>=budget:break
   pieces.append(g.loc[chosen])
  selected=pd.concat(pieces,ignore_index=True)
 return selected.sort_values(['timestamp','station_id','link_id']).reset_index(drop=True)
def load_runtime(name):
 folder=d.ROOT/'outputs/overnight'/(name+'_candidate');sys.path.insert(0,str(folder));spec=importlib.util.spec_from_file_location('fresh_'+name+'_apply',folder/'apply_residual.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 routes=e.predictor.load_routes();runtimes={name:load_runtime(name) for name in ['context','late55']};templates={};plan={}
 for panel in routes:
  k=pd.read_csv(d.DATA/'task1'/panel/'train/sample_submission_state.csv',dtype=str,usecols=d.KEYS);templates[panel]=k;v=k.assign(date=k.timestamp.str[:10]);v=v[(v.date>='2031-02-05')&(v.date<='2031-02-25')];plan[panel]=sorted(set(v.groupby('mask_regime').date.min())|set(v.groupby('mask_regime').date.max()));assert len(plan[panel])==6
 protocol=dict(scope='Fresh additional pseudo-target masks on exposed public train observations; NOT whole-system blind or official evaluation',seed=SEED,conditions=CONDITIONS,dates=plan,variants=['baseline','context','late55'],target_selection='10% of finite eligible currently observed cells; no value-based selection',blanked=['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm'],source_sha256=d.sha(__file__),feature_caches='Original-mask feature caches forbidden; recompute from RAM-masked inputs',no_raw_files_modified=True,no_kaggle_upload=True)
 if (O/'protocol.json').exists():assert json.loads((O/'protocol.json').read_text())==protocol
 else:d.save(O/'protocol.json',protocol)
 reader=pd.read_parquet;rows=[]
 for panel,route in routes.items():
  fd=pd.read_csv(d.DATA/'corridors'/panel/'network/fd_parameters.csv',dtype={'link_id':str}).groupby('link_id').mean(numeric_only=True);length=pd.read_csv(d.DATA/'corridors'/panel/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id').length_km;k=templates[panel];loaded={name:runtimes[name].load_panel(panel) for name in runtimes}
  for condition in CONDITIONS:
   d.state(status='running',stage='fresh_mask_robustness',panel=panel,condition=condition,next='Resume fresh_masks.py under shared lock; original-mask caches must not be used')
   record=O/(panel+'_'+condition+'.json')
   if record.exists():rows.extend(json.loads(record.read_text()));continue
   items=[];outputs={v:[] for v in ['baseline','context','late55']}
   for date in plan[panel]:
    d.check_deadline();case=O/'cases'/panel/condition/date;case.mkdir(parents=True,exist_ok=True);mpath=next((d.DATA/'corridors'/panel/'train/mainline_states_masked').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));raw=reader(mpath);raw.station_id=raw.station_id.astype(str);raw.link_id=raw.link_id.astype(str);raw.timestamp=raw.timestamp.astype(str);extra=mask_selection(raw,panel,date,condition);original=k[k.timestamp.str[:10]==date].copy();reg=original.mask_regime.iloc[0];extra['panel']=panel;extra['mask_regime']=reg;extra=extra[d.KEYS];assert not extra.merge(original,on=d.KEYS).shape[0]
    extra.to_csv(case/'auxiliary_keys.csv',index=False);union=pd.concat([original,extra],ignore_index=True).sort_values(d.KEYS).reset_index(drop=True);assert not union.duplicated(d.KEYS).any();idx=pd.MultiIndex.from_frame(raw[['timestamp','station_id','link_id']]);chosen=pd.MultiIndex.from_frame(extra[['timestamp','station_id','link_id']]);aug=raw.copy();hide=idx.isin(chosen);assert hide.sum()==len(extra)
    for col in protocol['blanked']:
     if col in aug:aug.loc[hide,col]=np.nan
    d.save(case/'mask_manifest.json',dict(raw_sha256=d.sha(mpath),keys_sha256=d.sha(case/'auxiliary_keys.csv'),original_targets=len(original),auxiliary_targets=len(extra),seed=SEED,labels_read_for_inference=False))
    e.predictor.keys_for=lambda data,road,split,which_date:union.copy()
    def masked_read(path,*args,**kwargs):
     text=str(path).replace('\\','/')
     if '/mainline_states/' in text:raise RuntimeError('Unmasked truth read during inference')
     if Path(path).resolve()==mpath.resolve():
      frame=aug.copy();cols=kwargs.get('columns',args[0] if args else None)
      return frame if cols is None else frame[list(cols)].copy()
     return reader(path,*args,**kwargs)
    pd.read_parquet=masked_read
    try:
     pred=e.predictor.predict(panel,date,route,d.DATA,case/'baseline','cuda');assert pred[d.KEYS].reset_index(drop=True).equals(union[d.KEYS]);all_predictions={'baseline':pred[d.KEYS+['speed_kmh','flow_vph']].copy()}
     for name,m in runtimes.items():
      policy,models=loaded[name];all_predictions[name]=m.correct(d.DATA,panel,date,pred,'train',policy,models)
    finally:pd.read_parquet=reader
    # Public truth is accessed only after all model outputs are complete.
    truthpath=next((d.DATA/'corridors'/panel/'train/mainline_states').rglob('synthetic_mainline_'+date.replace('-','_')+'.parquet'));truth=reader(truthpath,columns=['timestamp','station_id','link_id','speed_kmh','flow_vph']);truth[['timestamp','station_id','link_id']]=truth[['timestamp','station_id','link_id']].astype(str);truth=truth.rename(columns={'speed_kmh':'true_v','flow_vph':'true_q'})
    frame=None
    for name,p in all_predictions.items():
     selected=extra.merge(p,on=d.KEYS,validate='one_to_one');assert len(selected)==len(extra);selected.to_parquet(case/(name+'_auxiliary_predictions.parquet'),index=False);values=selected[['speed_kmh','flow_vph']].to_numpy();assert np.isfinite(values).all() and (values>=0).all();outputs[name].append(values)
     if name=='baseline':frame=selected.merge(truth,on=['timestamp','station_id','link_id'],validate='one_to_one')
    frame['lanes']=frame.link_id.map(fd.lanes);frame['length_km']=frame.link_id.map(length);frame['q_lane']=frame.flow_vph/frame.lanes;frame['true_q_lane']=frame.true_q/frame.lanes;assert frame[['true_v','true_q','lanes','length_km']].notna().all().all();items.append(dict(date=date,frame=frame));print('fresh',panel,condition,date,len(extra),flush=True)
   rr=[dict(panel=panel,condition=condition,variant=name,**e.metrics(items,pp)) for name,pp in outputs.items()];rows.extend(rr);d.save(record,rr);pd.DataFrame(rows).to_csv(O/'by_panel.csv',index=False);print('FRESH RESULT',panel,condition,rr,flush=True)
 summary=pd.DataFrame(rows).groupby(['condition','variant'])[['S_state','speed_RMSE','flow_per_lane_RMSE','station_target_deltaN_RMSE']].mean();summary.to_csv(O/'summary.csv');print(summary.to_string(),flush=True);d.state(status='stage_complete',stage='fresh_mask_robustness_complete',next='Assess genuinely recomputed pseudo-mask robustness; register next hypothesis before tuning. No official or blind-test claims, no Kaggle uploads.')
if __name__=='__main__':
 lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 try:main()
 except Exception as err:d.state(status='failed_or_cutoff',stage='fresh_mask_robustness',error=repr(err),next='Inspect fresh_masks.log; fix benchmark implementation without changing selected targets or frozen models');traceback.print_exc();raise
 finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()
