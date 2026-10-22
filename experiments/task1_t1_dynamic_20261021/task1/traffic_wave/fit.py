from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import lightgbm as lgb

from ..cache import snapshot
from ..cache_builder import RESIDUAL_COLUMNS
from ..data import KEY,save_json
from ..pipeline import load_booster,model_parameters,save_booster


def ordered_key_hash(keys):
    if keys[KEY].isna().any().any() or keys.duplicated(KEY).any():raise ValueError('Invalid training keys')
    return hashlib.sha256(pd.util.hash_pandas_object(keys[KEY],index=False).values.tobytes()).hexdigest()


def fit_models(cache,features,training,labels,keys,days,source_manifest,cfg):
    if len(features)!=len(keys) or labels.shape!=(len(keys),2):raise ValueError('Training features/labels/keys are not aligned')
    if len(keys)==0 or keys.panel.nunique()!=1:raise ValueError('One nonempty panel is required')
    if any(column.startswith('__') for column in features):raise ValueError('Identifiers or labels must not be model features')
    if set(keys.timestamp.str[:10])-set(source_manifest['metadata']['fit_dates']):raise ValueError('Non-training dates entered the fit matrix')
    params=model_parameters(cfg)
    identity=dict(kind='traffic_wave_enhanced_models',schema='traffic_wave_v1',panel=str(keys.panel.iloc[0]),
        source_training_key=source_manifest['key'],source_training_artifacts=source_manifest['artifacts'],
        ordered_keys_sha256=ordered_key_hash(keys),fit_dates=sorted(keys.timestamp.str[:10].unique()),
        wave_days=[dict(date=d['date'],key=d['key']) for d in days],features=list(features.columns),
        feature_matrix_sha256=hashlib.sha256(features.to_numpy(dtype=np.float32).tobytes()).hexdigest(),
        residual_targets_sha256=hashlib.sha256(training[RESIDUAL_COLUMNS].to_numpy(dtype=float).tobytes()).hexdigest(),
        parameters=params,rounds=cfg['rounds'],exact_direct_labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        implementation=snapshot([Path(__file__),Path(__file__).parents[1]/'pipeline.py']),
        packages={p:importlib.metadata.version(p) for p in ['lightgbm','numpy','pandas']})
    def build(destination):
        residual=training[RESIDUAL_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(residual).all() or not np.isfinite(labels).all():raise ValueError('Non-finite training labels')
        for mode,target in [('residual',residual),('direct',labels)]:
            for ch,name in enumerate(['speed','flow_lane']):
                model=lgb.train(params,lgb.Dataset(features,label=target[:,ch],free_raw_data=True),num_boost_round=cfg['rounds'])
                save_booster(model,destination/f'{mode}_{name}.txt')
                pd.DataFrame(dict(feature=model.feature_name(),gain=model.feature_importance('gain'))).sort_values(
                    'gain',ascending=False).to_csv(destination/f'{mode}_{name}_importance.csv',index=False)
        metadata=dict(training_rows=len(keys),features=list(features.columns),parameters=params,rounds=cfg['rounds'],
            ordered_keys_sha256=identity['ordered_keys_sha256'],original_training_key=source_manifest['key'],
            label_units=['km/h','veh/h/lane'],no_in_sample_model_predictions_used=True,
            same_rows_features_and_parameters_for_residual_and_direct=True)
        save_json(destination/'model_config.json',metadata)
        return metadata
    return cache.get_or_build(identity,build)


def load_models(folder):
    return {mode:[load_booster(folder/f'{mode}_{ch}.txt') for ch in ['speed','flow_lane']] for mode in ['residual','direct']}


def enhanced_values(features,initial,gap_minutes,models,lanes,grin_values,threads):
    residual=np.maximum(initial+np.column_stack([m.predict(features,num_threads=threads) for m in models['residual']]),0)
    direct=np.maximum(np.column_stack([m.predict(features,num_threads=threads) for m in models['direct']]),0)
    gap=np.asarray(gap_minutes,float)
    use_direct=~np.isfinite(gap)|(gap>60)
    result=np.where(use_direct[:,None],direct,.5*(residual+direct))
    result[:,1]*=np.asarray(lanes,float)
    use_grin=np.isfinite(gap)&(gap>60)
    result=np.where(use_grin[:,None],grin_values,result)
    return result,use_grin
