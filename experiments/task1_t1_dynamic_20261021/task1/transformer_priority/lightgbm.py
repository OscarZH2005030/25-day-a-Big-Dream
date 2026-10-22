from __future__ import annotations

from pathlib import Path
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
from .common import HERE, SCALE, read, save_json, versions, snapshot


def write_booster(model,path):
    # LightGBM's native file API cannot open some Windows Unicode paths.
    Path(path).write_text(model.model_to_string(num_iteration=model.best_iteration),encoding='utf-8')


def read_booster(path):
    return lgb.Booster(model_str=Path(path).read_text(encoding='utf-8'))


def fit(data,smoke,threads):
    # Same cached float32-normalized target precision as the original tree pipeline.
    for purpose in ['training','internal']:
        for date in data.roles[purpose]:
            data.labels(date,purpose)
    params=dict(objective='regression',metric='rmse',learning_rate=.05,num_leaves=31,min_data_in_leaf=80,
                lambda_l2=1.,feature_fraction=.9,num_threads=threads,seed=2026,deterministic=True,
                force_col_wise=True,verbosity=-1,device_type='cpu')
    rounds,patience=(50,10) if smoke else (2000,100)
    dates=data.roles['training']+data.roles['internal']
    identity=dict(kind='tt_shared_lightgbm',panel=data.panel,dates=data.roles,features=[data.records[d]['key'] for d in dates],
                  labels=[data.label_records[d]['key'] for d in dates],parameters=params,rounds=rounds,patience=patience,
                  lightgbm_version=versions()['lightgbm'],implementation=snapshot([HERE/'lightgbm.py']),
                  label_precision='original float32 normalized then restored',schema=1)
    def build(out):
        scratch=out/'scratch';scratch.mkdir();arrays={}
        for purpose in ['training','internal']:
            ds=data.roles[purpose];n=sum(data.records[d]['rows'] for d in ds)
            x=np.lib.format.open_memmap(scratch/(purpose+'_x.npy'),mode='w+',dtype=np.float32,shape=(n,51))
            y=np.lib.format.open_memmap(scratch/(purpose+'_y.npy'),mode='w+',dtype=np.float64,shape=(n,2))
            cursor=0
            for date in ds:
                features,base=data.target_arrays(date);end=cursor+len(features)
                x[cursor:end]=features
                y[cursor:end]=(data.labels(date,purpose)/SCALE).astype(np.float32).astype(np.float64)*SCALE-base
                cursor=end
            x.flush();y.flush();arrays[purpose]=(x,y)
        choices=[]
        for c,name in enumerate(['speed_kmh','flow_per_lane']):
            print('[LightGBM]',data.panel,name,'training_targets',len(arrays['training'][0]),flush=True)
            training=lgb.Dataset(arrays['training'][0],label=np.asarray(arrays['training'][1][:,c]).copy(),feature_name=data.names)
            validation=lgb.Dataset(arrays['internal'][0],label=np.asarray(arrays['internal'][1][:,c]).copy(),reference=training)
            trace={}
            model=lgb.train(params,training,num_boost_round=rounds,valid_sets=[validation],valid_names=['internal'],
                            callbacks=[lgb.early_stopping(patience,first_metric_only=True),lgb.record_evaluation(trace),lgb.log_evaluation(100)])
            write_booster(model,out/(name+'.txt'))
            save_json(out/(name+'_trace.json'),trace)
            choices.append(dict(channel=name,best_iteration=model.best_iteration,internal_RMSE=float(model.best_score['internal']['rmse'])))
            del model,training,validation
            gc.collect()
        # Close memory maps before deleting the four known temporary files on Windows.
        for pair in arrays.values():
            for value in pair:
                value._mmap.close()
        arrays.clear();gc.collect()
        for path in list(scratch.iterdir()):
            if path.name not in ['training_x.npy','training_y.npy','internal_x.npy','internal_y.npy']:
                raise ValueError('Unexpected scratch artifact')
            path.unlink()
        # Empty directories are not cache artifacts and need no cleanup operation.
        return dict(choices=choices,training_rows=sum(data.records[d]['rows'] for d in data.roles['training']))
    folder,manifest,hit=data.cache.get_or_build(identity,build)
    models=[read_booster(folder/(name+'.txt')) for name in ['speed_kmh','flow_per_lane']]
    return models,dict(key=manifest['key'],folder=str(folder),hit=hit,metadata=manifest['metadata'])


def predictions(data,models,record,dates,threads):
    identity=dict(kind='tt_lightgbm_predictions',panel=data.panel,model=record['key'],days=[data.records[d]['key'] for d in dates],schema=1)
    def build(out):
        values=[]
        for date in dates:
            x,base=data.target_arrays(date)
            values.append(np.maximum(base+np.column_stack([m.predict(x,num_threads=threads) for m in models]),0))
        values=np.concatenate(values);keys=data.expected(dates);ii=np.concatenate([data.array(d,'ii') for d in dates])
        keys['speed_kmh']=values[:,0];keys['flow_vph']=values[:,1]*data.exp.lanes[ii]
        keys.to_parquet(out/'predictions.parquet',index=False)
        return dict(rows=len(keys),dates=dates)
    folder,manifest,hit=data.cache.get_or_build(identity,build)
    return pd.read_parquet(folder/'predictions.parquet'),dict(key=manifest['key'],folder=str(folder),hit=hit)
