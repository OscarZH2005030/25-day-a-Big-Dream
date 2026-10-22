from __future__ import annotations

import json
import platform
import time
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
import torch

from ..cache import FeatureCache,snapshot
from ..cache_builder import IDENTIFIER_COLUMNS
from ..data import KEY,TARGETS,day_files,network,save_json
from ..features import feature_columns
from ..fusion_experiments import aligned_prediction
from ..grin.data import key_positions
from ..grin.hybrid import check_scores
from ..grin.runner import comparison
from ..metrics import aggregate,score,validate_predictions
from ..pipeline import panel_families,truth_rows
from .cache import day_cache,kernel_cache,make_engine
from .core import DEFAULTS,row_features,validate_wave
from .fit import enhanced_values,fit_models,load_models,ordered_key_hash
from .log import append_log
from .reference import load_sources

PROJECT=Path(__file__).resolve().parents[2]
CURRENT,WAVE,ENHANCED='Current_best_hybrid','Traffic_wave','Wave_enhanced_hybrid'


def validate(cfg):
    if cfg['panel']!='D12_I5_N' or cfg['seed']!=20261016:raise ValueError('This experiment fixes I5 and the original seed')
    if cfg['threads']<1:raise ValueError('Threads must be positive')
    validate_wave(cfg)


def resource_usage(device):
    if device.type!='cuda':return dict(device='CPU',measured=False)
    return dict(device=torch.cuda.get_device_name(device),measured=True,
        total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        note='Current process PyTorch allocator; not whole-device consumption')


def reference_check(source,current,truth,net,family,dates,smoke):
    rows,_=score(current,truth,net,family,'Baseline_GRIN_gap60')
    old=source['reference']
    if smoke:
        daily=pd.read_csv(Path(source['hybrid_path'])/'Baseline_GRIN_gap60_daily_errors.csv')
        daily=daily.loc[daily.date.isin(dates)];reference=[]
        for regime,group in daily.groupby('mask_regime'):
            rv=float(np.sqrt(np.average(group.speed_mse,weights=group.n)))
            rq=float(np.sqrt(np.average(group.flow_lane_mse,weights=group.n)))
            reference.append(dict(method='Baseline_GRIN_gap60',panel='D12_I5_N',regime=regime,n=int(group.n.sum()),
                rmse_speed_kmh=rv,rmse_flow_per_lane=rq,S_state=.54*max(0,1-rv/25)+.46*max(0,1-rq/600)))
        old=pd.DataFrame(reference)
    result=check_scores(old,pd.DataFrame(rows),'Baseline_GRIN_gap60')
    if not result['passed']:raise ValueError('Saved current-best scores did not reproduce')
    return result


def make_features(keys,base,grid_folder,grid,nodes,cache,kernel,engine):
    keys=keys.reset_index(drop=True);base=base.reset_index(drop=True)
    if len(keys)!=len(base):raise ValueError('Feature/key length mismatch')
    grouped=keys.groupby(keys.timestamp.str[:10],sort=True)
    extras=None;standalone=np.empty((len(keys),2));records=[];columns=None
    for date,part in grouped:
        indices=part.index.to_numpy();rr,cc=key_positions(part,nodes)
        with np.load(grid_folder/(date+'.npz'),allow_pickle=False) as archive:
            if archive['mask'][rr,cc].any():raise ValueError('A reconstruction target is visible in the source grid')
        reconstructed,record=day_cache(cache,date,grid_folder,grid,kernel,engine)
        features,values=row_features(rr,cc,reconstructed,base.iloc[indices])
        if columns is None:columns=list(features);extras=np.empty((len(keys),len(columns)),np.float32)
        if columns!=list(features):raise ValueError('Wave feature schema changed between dates')
        extras[indices]=features.to_numpy();standalone[indices]=values;records.append(record)
        if len(records)%10==0:print(f'[traffic wave] {len(records)}/{len(grouped)} days',flush=True)
    augmented=pd.concat([base,pd.DataFrame(extras,columns=columns)],axis=1)
    if any(c.startswith('__') for c in augmented):raise ValueError('Metadata entered augmented features')
    return augmented,standalone,records


def export_errors(output,method,pred,truth,net,family,diagnostics):
    rows,cells=score(pred,truth,net,family,method)
    cells=cells.drop(columns=['gap','selected_model'],errors='ignore').merge(diagnostics[KEY+['gap']],on=KEY,validate='one_to_one')
    for label,groups in [('daily',['date','mask_regime']),('gap',['gap']),('congestion',['congested'])]:
        cells.groupby(groups,as_index=False).agg(n=('speed_sq_error','size'),speed_mse=('speed_sq_error','mean'),
            flow_lane_mse=('flow_lane_sq_error','mean')).to_csv(output/f'{method}_{label}_errors.csv',index=False)
    return rows


def run(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new, empty output directory')
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(DEFAULTS,panel='D12_I5_N',seed=20261016,sample_per_day=1000,
        fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',eval_end='2031-01-31',
        rounds=50 if args.smoke else 350,extra_features='Traffic wave only; no previous fitted predictions')
    validate(cfg);started=time.perf_counter()
    torch.set_num_threads(cfg['threads']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    if cfg['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run setup_traffic_wave.ps1 or explicitly choose CPU')
    device=torch.device('cuda' if cfg['device']=='cuda' or (cfg['device']=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    print(f'[traffic wave] I5; device={device}; rounds={cfg["rounds"]}; mode={"smoke" if args.smoke else "formal"}',flush=True)
    source=load_sources(cfg);source['hybrid_path']=cfg['hybrid_run']
    train_dates=source['grid_train']['metadata']['dates'][:14] if args.smoke else source['grid_train']['metadata']['dates']
    eval_dates=source['grid_eval']['metadata']['dates'][:3] if args.smoke else source['grid_eval']['metadata']['dates']
    use=source['keys'].timestamp.str[:10].isin(train_dates).to_numpy()
    training=source['train'].loc[use].reset_index(drop=True);train_keys=source['keys'].loc[use].reset_index(drop=True)
    labels=source['labels'][use]
    expected=source['expected'].loc[source['expected'].timestamp.str[:10].isin(eval_dates)].reset_index(drop=True)
    current=aligned_prediction(expected,source['current'].loc[source['current'].timestamp.str[:10].isin(eval_dates)])
    grin=aligned_prediction(expected,source['grin'].loc[source['grin'].timestamp.str[:10].isin(eval_dates)])
    diagnostics=expected.merge(source['diagnostics'],on=KEY,validate='one_to_one')
    root=Path(cfg['release_root']);net=network(root,cfg['panel']);family=panel_families(root)[cfg['panel']]
    truth=truth_rows(root,cfg['panel'],expected)
    check=reference_check(source,current,truth,net,family,eval_dates,args.smoke)
    cache=FeatureCache(Path(cfg['cache_dir']));nodes=pd.DataFrame(source['grid_train']['metadata']['nodes'])
    kernel_folder,kernel,kernel_hit=kernel_cache(cache,root,cfg['panel'],nodes,net,cfg)
    output.mkdir(parents=True,exist_ok=True)
    cfg.update(actual_fit_dates=train_dates,actual_eval_dates=eval_dates)
    save_json(output/'run_config.json',cfg)
    save_json(output/'reference_check.json',dict(passed=True,check=check,subset=args.smoke,
        note='Saved full-budget model on the same target subset' if args.smoke else 'Complete January current-best control reproduced'))
    sample_check=dict(passed=True,formal_pool_rows=len(source['keys']),used_rows=len(train_keys),
        exact_ordered_keys_equal_to_baseline_and_grin=True,ordered_keys_sha256=ordered_key_hash(train_keys),
        no_previous_fitted_prediction_features=True)
    save_json(output/'sample_check.json',sample_check)
    environment=dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['torch','lightgbm','numpy','pandas','pyarrow']},
        source_files=snapshot(list(Path(__file__).parent.glob('*.py'))),device=str(device),kernel_key=kernel['key'],
        adaptation='ASM-inspired; fixed graph distances, per-channel masks, within-day support; not the original ASM software')
    save_json(output/'environment.json',environment)
    usage=dict(kernel=dict(key=kernel['key'],hit=kernel_hit),original_feature_cache_reused=True,original_grid_cache_reused=True,
        original_grin_retrained=False,original_control_predictions_reused=True,training_days=[],evaluation_days=[])
    if args.action=='prepare':
        save_json(output/'cache_usage.json',usage);save_json(output/'log_update.json',dict(status='skipped_prepare'))
        print(f'Preflight passed: {len(train_keys)} fit targets, {len(expected)} eval targets, {len(nodes)} nodes; no new model trained. {output}',flush=True);return
    engine=make_engine(kernel_folder,kernel,cfg,device)
    base_columns=feature_columns(source['training']['metadata']['feature_columns'],'multihop')
    x_train,_,train_days=make_features(train_keys,training[base_columns],source['grid_train_folder'],source['grid_train'],nodes,cache,kernel,engine)
    usage['training_days']=train_days;save_json(output/'cache_usage.json',usage)
    model_folder,models_meta,model_hit=fit_models(cache,x_train,training,labels,train_keys,train_days,source['training'],cfg)
    usage['enhanced_models']=dict(key=models_meta['key'],hit=model_hit,training_rows=len(train_keys),feature_count=len(x_train.columns))
    save_json(output/'model_cache_reference.json',dict(folder=str(model_folder),key=models_meta['key'],metadata=models_meta['metadata']))
    models=load_models(model_folder);del x_train
    columns=list(dict.fromkeys(base_columns+IDENTIFIER_COLUMNS+['__initial_speed','__initial_flow_lane','__history_speed','__history_flow_lane','__lanes']))
    tables=[pd.read_parquet(source['eval_folder']/(day+'.parquet'),columns=columns) for day in eval_dates]
    evaluation=pd.concat(tables,ignore_index=True);del tables
    eval_keys=evaluation[IDENTIFIER_COLUMNS].rename(columns={key:key[2:] for key in IDENTIFIER_COLUMNS})
    # Reorder metadata to the original full natural-key template, never positional coincidence.
    order=expected.merge(eval_keys.assign(__row=np.arange(len(eval_keys))),on=KEY,validate='one_to_one')['__row'].to_numpy(int)
    if len(order)!=len(eval_keys):raise ValueError('Evaluation feature scope differs')
    evaluation=evaluation.iloc[order].reset_index(drop=True)
    x_eval,wave_values,eval_days=make_features(expected,evaluation[base_columns],source['grid_eval_folder'],source['grid_eval'],nodes,cache,kernel,engine)
    usage['evaluation_days']=eval_days;save_json(output/'cache_usage.json',usage)
    initial=evaluation[['__initial_speed','__initial_flow_lane']].to_numpy(float);lanes=evaluation['__lanes'].to_numpy(float)
    # Use the exact cached historical fallback for standalone output when unsupported.
    for ch,name in enumerate(['speed','flow_lane']):
        fallback=x_eval[f'wave_{name}_fallback'].to_numpy(bool)
        wave_values[fallback,ch]=evaluation.loc[fallback,'__history_'+name].to_numpy(float)
    wave=expected.copy();wave_values[:,1]*=lanes;wave[TARGETS]=np.maximum(wave_values,0)
    enhanced=expected.copy()
    values,gate=enhanced_values(x_eval,initial,diagnostics.gap_minutes,models,lanes,grin[TARGETS].to_numpy(float),cfg['threads'])
    enhanced[TARGETS]=values;enhanced['selected_model']=np.where(gate,'original_GRIN','wave_augmented_LightGBM')
    if not np.array_equal(enhanced.loc[gate,TARGETS].to_numpy(),grin.loc[gate,TARGETS].to_numpy()):
        raise ValueError('Long-gap GRIN preservation failed')
    predictions={CURRENT:current,WAVE:wave,ENHANCED:enhanced}
    rows=[]
    for method,pred in predictions.items():
        validate_predictions(expected,pred);pred.to_parquet(output/f'{method}_predictions.parquet',index=False)
        rows+=export_errors(output,method,pred,truth,net,family,diagnostics)
    detail=pd.DataFrame(rows);table=comparison(detail);control=table.loc[table.method.eq(CURRENT)].iloc[0]
    table['delta_S_state_vs_current_best']=table.S_state-control.S_state
    table['delta_speed_rmse_vs_current_best']=table.rmse_speed_kmh-control.rmse_speed_kmh
    table['delta_flow_rmse_vs_current_best']=table.rmse_flow_per_lane-control.rmse_flow_per_lane
    detail.to_csv(output/'metrics_by_panel_regime.csv',index=False);aggregate(detail).to_csv(output/'metrics_summary.csv',index=False)
    table.to_csv(output/'comparison_methods.csv',index=False)
    coverage=diagnostics[KEY+['gap']].copy()
    for name in ['speed','flow_lane']:
        coverage[f'{name}_fallback']=x_eval[f'wave_{name}_fallback'].to_numpy()
        coverage[f'{name}_coverage_free']=x_eval[f'wave_{name}_coverage_free'].to_numpy()
        coverage[f'{name}_coverage_congested']=x_eval[f'wave_{name}_coverage_congested'].to_numpy()
    coverage.groupby(['mask_regime','gap'],as_index=False).agg(n=('speed_fallback','size'),
        speed_fallback=('speed_fallback','sum'),flow_fallback=('flow_lane_fallback','sum'),
        speed_coverage_free=('speed_coverage_free','mean'),speed_coverage_congested=('speed_coverage_congested','mean'),
        flow_coverage_free=('flow_lane_coverage_free','mean'),flow_coverage_congested=('flow_lane_coverage_congested','mean')).to_csv(output/'wave_coverage_by_gap.csv',index=False)
    importance=[]
    for mode in ['residual','direct']:
        for ch in ['speed','flow_lane']:
            imp=pd.read_csv(model_folder/f'{mode}_{ch}_importance.csv');imp['target_mode']=mode;imp['channel']=ch;importance.append(imp)
    pd.concat(importance,ignore_index=True).to_csv(output/'feature_importance.csv',index=False)
    save_json(output/'memory.json',resource_usage(device))
    save_json(output/'integrity_check.json',dict(passed=True,all_targets_matched=True,long_gap_equals_saved_GRIN=True,
        long_gap_rows=int(gate.sum()),no_previous_fitted_prediction_features=True,source_models_retrained=False,
        exact_original_label_sidecar=True,new_feature_count=len(x_eval.columns)-len(base_columns)))
    own_manifests=[Path(cfg['cache_dir'])/r['key']/'manifest.json' for r in train_days+eval_days]
    full=day_files(root,cfg['panel'],'train',False)
    save_json(output/'source_manifest.json',snapshot(source['provenance']+[kernel_folder/'manifest.json',model_folder/'manifest.json']+
        own_manifests+[full[day] for day in eval_dates]+list(Path(__file__).parent.glob('*.py'))))
    save_json(output/'COMPLETE.json',dict(experiment='traffic_wave',panel=cfg['panel'],smoke=args.smoke,methods=list(predictions),
        training_rows=len(train_keys),evaluation_rows=len(expected),reference_passed=True,pairing_passed=True,
        integrity_passed=True,seconds=time.perf_counter()-started))
    try:status=append_log(output,PROJECT/'实验日志.txt')
    except (OSError,ValueError) as error:
        save_json(output/'log_update.json',dict(status='failed',error=str(error)))
        raise RuntimeError('Wave experiment completed; retry log action without training') from error
    print(table.to_string(index=False),flush=True);print(f'Memory: {resource_usage(device)}',flush=True)
    print(f'Log status: {status["status"]}. Traffic wave complete: {output}',flush=True)
