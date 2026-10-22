from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import torch

from ..cache import snapshot
from ..cache_builder import IDENTIFIER_COLUMNS
from ..data import KEY,TARGETS,read_csv,save_json
from ..features import feature_columns
from ..fusion_experiments import aligned_prediction
from ..metrics import validate_predictions
from ..traffic_wave.cache import kernel_cache,make_engine,day_cache,backend_identity
from ..traffic_wave.fit import fit_models,load_models,enhanced_values,ordered_key_hash
from ..traffic_wave.runner import make_features
from ..traffic_wave.reference import verified_manifest
from ..wave_neural.cache import NeuralCache
from ..wave_neural.data import training_statistics,physical_cache,input_day,ramp_files
from ..wave_context.data import ContextWindows
from ..wave_context.model import receptive_field_check
from ..wave_context.runner import train_model,initialization_check
from ..speed_fusion.rules import CONTROL,NEURAL
from .reference import PROJECT,PANEL


def implementation():
    files=list(Path(__file__).parent.glob('*.py'))
    for package in ['traffic_wave','wave_neural','wave_context','speed_fusion','grin']:
        files+=list((PROJECT/'task1'/package).glob('*.py'))
    files += [PROJECT/'task1'/name for name in ['cache.py','cache_builder.py','data.py','features.py','fusion_experiments.py','metrics.py','pipeline.py','target_labels.py']]
    return snapshot(files)


def common_identity(cfg,source,device):
    return dict(panel=PANEL,source_training_key=source['training']['key'],source_evaluation_key=source['evaluation']['key'],
        training_artifacts=source['training']['artifacts'],training_grid_key=source['grid_train']['key'],evaluation_grid_key=source['grid_eval']['key'],
        backend=backend_identity(device),packages={p:importlib.metadata.version(p) for p in ['torch','lightgbm','numpy','pandas','pyarrow']},
        implementation=implementation(),seed=cfg['seed'],threads=cfg['threads'])


def cached_prediction(cache,identity,expected,builder):
    def build(destination):
        prediction,metadata=builder(destination)
        prediction=aligned_prediction(expected,prediction)
        prediction.to_parquet(destination/'predictions.parquet',index=False)
        return dict(metadata,evaluation_rows=len(expected))
    folder,manifest,hit=cache.get_or_build(identity,build)
    prediction=aligned_prediction(expected,pd.read_parquet(folder/'predictions.parquet'))
    if manifest['metadata']['evaluation_rows']!=len(expected):raise ValueError('Prediction cache target scope differs')
    return prediction,dict(key=manifest['key'],folder=str(folder),hit=hit,metadata=manifest['metadata'])


def resources(source,cfg,nodes,net):
    wave_cache=NeuralCache(PROJECT/'cache/traffic_wave');input_cache=NeuralCache(PROJECT/'cache/wave_neural')
    kernel_folder,kernel,kernel_hit=kernel_cache(wave_cache,source['root'],PANEL,nodes,net,cfg)
    stats_folder,stats,stats_hit=training_statistics(input_cache,source['grid_train_folder'],source['grid_train'])
    attrs_folder,attrs,attrs_hit=physical_cache(input_cache,source['root'],PANEL,nodes)
    means=np.load(stats_folder/'means.npy',allow_pickle=False);attributes=np.load(attrs_folder/'attributes.npy',allow_pickle=False)
    source['stats']=stats;source['attributes_manifest']=attrs
    return dict(wave_cache=wave_cache,input_cache=input_cache,kernel_folder=kernel_folder,kernel=kernel,kernel_hit=kernel_hit,
        stats_folder=stats_folder,stats=stats,stats_hit=stats_hit,attrs_folder=attrs_folder,attrs=attrs,attrs_hit=attrs_hit,
        means=means,attributes=attributes,ramps=ramp_files(source['root'],PANEL),
        attachments=read_csv(source['root']/'corridors'/PANEL/'network/ramp_attachment_map.csv'))


def check_grin_preservation(prediction,grin,minutes):
    gate=np.isfinite(minutes)&(np.asarray(minutes)>60)
    if not np.array_equal(prediction[TARGETS].to_numpy()[gate],grin[TARGETS].to_numpy()[gate]):
        raise ValueError('Wave-enhanced long gaps changed the saved GRIN')
    return dict(passed=True,long_gap_rows=int(gate.sum()),both_channels_equal_saved_GRIN=True)


def wave_predictions(cache,cfg,source,res,nodes,train_keys,expected,diagnostics,grin,train_dates,eval_dates,device,events):
    identity=dict(common_identity(cfg,source,device),kind='d7_wave_predictions',schema='frozen_i5_wave_recipe_v1',
        wave_parameters={k:cfg[k] for k in source['recipe']['wave_parameters']},kernel_key=res['kernel']['key'],rounds=cfg['rounds'],
        training_keys_sha256=ordered_key_hash(train_keys),evaluation_keys_sha256=ordered_key_hash(expected),
        direct_labels_sha256=hashlib.sha256(source['labels'].tobytes()).hexdigest(),train_dates=train_dates,eval_dates=eval_dates)
    def build(destination):
        events['wave_prediction_inference']=True
        engine=make_engine(res['kernel_folder'],res['kernel'],cfg,device)
        use=source['keys'].timestamp.str[:10].isin(train_dates).to_numpy()
        training=source['train'].loc[use].reset_index(drop=True);labels=source['labels'][use]
        columns=feature_columns(source['training']['metadata']['feature_columns'],'multihop')
        x_train,_,train_days=make_features(train_keys,training[columns],source['grid_train_folder'],source['grid_train'],nodes,res['wave_cache'],res['kernel'],engine)
        model_folder,model,model_hit=fit_models(res['wave_cache'],x_train,training,labels,train_keys,train_days,source['training'],cfg)
        events['wave_models_trained']=not model_hit;events['wave_model_cache_accessed']=True
        models=load_models(model_folder);del x_train
        wanted=list(dict.fromkeys(columns+IDENTIFIER_COLUMNS+['__initial_speed','__initial_flow_lane','__history_speed','__history_flow_lane','__lanes']))
        filenames={Path(name).stem:name for name in source['evaluation']['metadata']['files']}
        evaluation=pd.concat([pd.read_parquet(source['eval_folder']/filenames[d],columns=wanted) for d in eval_dates],ignore_index=True)
        keys=evaluation[IDENTIFIER_COLUMNS].rename(columns={k:k[2:] for k in IDENTIFIER_COLUMNS})
        order=expected[KEY].merge(keys.assign(__row=np.arange(len(keys))),on=KEY,validate='one_to_one',sort=False)['__row'].to_numpy(int)
        if len(order)!=len(keys):raise ValueError('D7 wave evaluation scope differs')
        evaluation=evaluation.iloc[order].reset_index(drop=True)
        x_eval,_,eval_days=make_features(expected,evaluation[columns],source['grid_eval_folder'],source['grid_eval'],nodes,res['wave_cache'],res['kernel'],engine)
        values,_=enhanced_values(x_eval,evaluation[['__initial_speed','__initial_flow_lane']].to_numpy(float),diagnostics.gap_minutes,
            models,evaluation['__lanes'].to_numpy(float),grin[TARGETS].to_numpy(float),cfg['threads'])
        prediction=expected.copy();prediction[TARGETS]=values
        preserved=check_grin_preservation(prediction,grin,diagnostics.gap_minutes.to_numpy())
        events['wave_days_built']=sum(not r['hit'] for r in train_days+eval_days)
        metadata=dict(training_rows=len(train_keys),feature_count=len(x_eval.columns),train_days=train_days,eval_days=eval_days,
            enhanced_models=dict(key=model['key'],folder=str(model_folder),hit_on_prediction_build=model_hit),
            grin_preservation=preserved,recipe='I5 fixed wave-enhanced LightGBM + saved D7 GRIN',original_models_retrained=False)
        save_json(destination/'selection.json',metadata)
        return prediction,metadata
    prediction,record=cached_prediction(cache,identity,expected,build)
    check_grin_preservation(prediction,grin,diagnostics.gap_minutes.to_numpy())
    return prediction,record


def daily_inputs(source,res,nodes,wave_record,events):
    result={}
    for split,key in [('training','train'),('evaluation','eval')]:
        rows=wave_record['metadata']['train_days' if key=='train' else 'eval_days'];records=[]
        for row in rows:
            date=row['date'];folder=res['wave_cache'].root/row['key'];manifest=verified_manifest(folder)
            record=input_day(res['input_cache'],date,source['grid_'+key+'_folder'],source['grid_'+key],folder,manifest,
                res['stats_folder'],res['stats'],res['attrs'],res['ramps'][date],res['attachments'],nodes)
            records.append(record)
        result[split]=records
        print(f'[D7 neural input] {split}: {len(records)} days; hits={sum(r["hit"] for r in records)}',flush=True)
    events['input_days_built']=sum(not r['hit'] for rows in result.values() for r in rows)
    return result


def neural_checks(source,res,records,adjacency,cfg,device):
    initial=initialization_check(records,source,res['means'],res['attributes'],adjacency,cfg,device)
    field=receptive_field_check(adjacency,device,cfg['window'])
    if initial['parameters']!=38722 or field['actual_window_slots']!=48:raise ValueError('D7 architecture differs from frozen I5')
    return dict(passed=True,initialization=initial,receptive_field=field,no_wave=True,shared_temporal_interpolation_prior=True)


def neural_predictions(cache,cfg,source,res,inputs,adjacency,nodes,train_keys,expected,truth,net,family,truth_hashes,device,events):
    identity=dict(common_identity(cfg,source,device),kind='d7_neural_predictions',schema='frozen_i5_no_wave_recipe_v1',
        architecture={k:cfg[k] for k in ['hidden','layers','window','stride','epochs','steps_per_epoch','batch_size','patience','learning_rate']},
        statistics_key=res['stats']['key'],attributes_key=res['attrs']['key'],
        training_inputs=[r['key'] for r in inputs['training']],evaluation_inputs=[r['key'] for r in inputs['evaluation']],
        training_keys_sha256=ordered_key_hash(train_keys),evaluation_keys_sha256=ordered_key_hash(expected),development_truth=truth_hashes)
    def build(destination):
        events['neural_prediction_inference']=True;events['neural_model_cache_accessed']=True
        train_set=ContextWindows(inputs['training'],source['grid_train']['metadata'],res['means'],res['attributes'],False,True,cfg['window'],cfg['stride'])
        eval_set=ContextWindows(inputs['evaluation'],source['grid_eval']['metadata'],res['means'],res['attributes'],False,False,cfg['window'],cfg['stride'])
        prediction,model=train_model(NeuralCache(PROJECT/'cache/wave_context'),NEURAL,cfg,train_set,eval_set,source,adjacency,
            expected,truth,net,family,nodes,train_keys,truth_hashes,device)
        events['neural_model_trained']=not model['hit']
        history=pd.read_csv(Path(model['folder'])/'training_history.csv');history.to_csv(destination/'training_history.csv',index=False)
        save_json(destination/'selection.json',model)
        return prediction,dict(training_rows=len(train_keys),model=model,no_wave=True,prior='temporal interpolation',original_GRIN_retrained=False)
    return cached_prediction(cache,identity,expected,build)


def prepare_probe(cfg,source,res,nodes,device,events):
    date=source['grid_train']['metadata']['dates'][0]
    engine=make_engine(res['kernel_folder'],res['kernel'],cfg,device)
    _,row=day_cache(res['wave_cache'],date,source['grid_train_folder'],source['grid_train'],res['kernel'],engine)
    events['wave_days_built']+=int(not row['hit'])
    temporary=dict(metadata=dict(train_days=[row],eval_days=[]))
    inputs=daily_inputs(source,res,nodes,temporary,events)
    adjacency=np.load(source['grid_train_folder']/'adjacency.npy',allow_pickle=False)
    return neural_checks(source,res,inputs['training'],adjacency,cfg,device)
