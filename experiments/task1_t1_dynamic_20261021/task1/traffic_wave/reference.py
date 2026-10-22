from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..cache import FeatureCache,feature_implementation
from ..cache_builder import IDENTIFIER_COLUMNS
from ..data import KEY,TARGETS,sha256
from ..fusion_experiments import aligned_prediction
from ..grin.hybrid import load_baseline,require_compatible_grin,switch_predictions
from ..pipeline import model_parameters
from ..target_labels import aligned_direct_targets

PROJECT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_manifest(folder):
    manifest=read(folder/'manifest.json')
    return FeatureCache(folder.parent)._validate(folder,manifest['identity'])


def load_sources(cfg):
    bundle=load_baseline(cfg['baseline_run'],cfg['panel'])
    original=bundle['config']
    dates=dict(fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',eval_end='2031-01-31')
    if any(original[k]!=v for k,v in dates.items()) or original['sample_per_day']!=1000 or original['seed']!=20261016:
        raise ValueError('Original baseline protocol differs')
    if str(Path(original['release_root']).resolve())!=str(Path(cfg['release_root']).resolve()):raise ValueError('Release roots differ')
    record=next(r for r in read(Path(cfg['baseline_run'])/'cache_usage.json') if r['panel']==cfg['panel'])
    base_cache=Path(original['cache_dir']).resolve();train_folder=bundle['training_table'].parent
    training=verified_manifest(train_folder)
    if training['identity']['implementation']!=feature_implementation():raise ValueError('Original feature implementation changed')
    eval_folder=base_cache/record['evaluation_key'];evaluation=verified_manifest(eval_folder)
    label_folder=base_cache/record['direct_labels_key'];label_manifest=verified_manifest(label_folder)
    if label_manifest['identity']['parent_training_key']!=training['key']:raise ValueError('Direct-label cache belongs to a different training table')
    if label_manifest['identity']['builder_sha256']!=sha256(PROJECT/'task1/target_labels.py'):raise ValueError('Direct-label implementation changed')
    train=pd.read_parquet(bundle['training_table'])
    keys=train[IDENTIFIER_COLUMNS].rename(columns={c:c[2:] for c in IDENTIFIER_COLUMNS}).reset_index(drop=True)
    labels=aligned_direct_targets(train,pd.read_parquet(label_folder/'direct_targets.parquet'))
    grin=Path(cfg['grin_run']);grin_cfg=read(grin/'run_config.json')
    require_compatible_grin(grin,grin_cfg)
    for k,v in dates.items():
        if grin_cfg[k]!=v:raise ValueError('GRIN protocol differs')
    if grin_cfg['panel']!=cfg['panel'] or grin_cfg['sample_per_day']!=1000 or grin_cfg['seed']!=20261016:
        raise ValueError('GRIN sample protocol differs')
    usage=read(grin/'cache_usage.json');grid_cache=Path(grin_cfg['cache_dir'])
    grid_train_folder=grid_cache/usage['training_key'];grid_eval_folder=grid_cache/usage['evaluation_key']
    grid_train=verified_manifest(grid_train_folder);grid_eval=verified_manifest(grid_eval_folder)
    sampled=pd.read_parquet(grid_train_folder/'training_sample_keys.parquet')
    if not keys.equals(sampled[KEY].reset_index(drop=True)) or len(keys)!=214000:raise ValueError('Training target order differs from the original 214000 samples')
    if len(grid_train['metadata']['dates'])!=214 or len(grid_eval['metadata']['dates'])!=31:
        raise ValueError('The reference must contain the complete training and January grids')
    hybrid=Path(cfg['hybrid_run']);complete=read(hybrid/'COMPLETE.json');hc=read(hybrid/'run_config.json')
    if complete.get('experiment')!='grin_hybrid' or not complete.get('reference_passed') or hc.get('smoke'):
        raise ValueError('Current-best reference must be a completed formal hybrid')
    if Path(hc['baseline_run']).resolve()!=Path(cfg['baseline_run']).resolve() or Path(complete['grin_source']).resolve()!=grin.resolve():
        raise ValueError('Current-best source models differ')
    expected=pd.read_parquet(grid_eval_folder/'expected_keys.parquet')
    current=aligned_prediction(expected,pd.read_parquet(hybrid/'combined_predictions.parquet'))
    grin_pred=aligned_prediction(expected,pd.read_parquet(grin/'GRIN_predictions.parquet'))
    diagnostics=expected.merge(bundle['diagnostics'],on=KEY,validate='one_to_one')
    reconstructed=switch_predictions(expected,bundle['baseline'],grin_pred,diagnostics.gap_minutes)
    if not np.array_equal(current[TARGETS].to_numpy(),reconstructed[TARGETS].to_numpy()):raise ValueError('Current best does not reproduce from its saved constituent predictions')
    model_meta=read(Path(cfg['baseline_run'])/'models/multihop/direct'/cfg['panel']/'target_model_config.json')
    required_params=dict(model_meta['parameters'],num_threads=cfg['threads'])
    if model_parameters(cfg)!=required_params:raise ValueError('Enhanced LightGBM parameters differ from the frozen baseline')
    if model_meta['training_table_sha256']!=sha256(bundle['training_table']):raise ValueError('Original model training sample table differs')
    return dict(bundle=bundle,train=train,keys=keys,labels=labels,training=training,
        eval_folder=eval_folder,evaluation=evaluation,grid_train_folder=grid_train_folder,grid_train=grid_train,
        grid_eval_folder=grid_eval_folder,grid_eval=grid_eval,expected=expected,current=current,grin=grin_pred,
        diagnostics=diagnostics,reference=pd.read_csv(hybrid/'metrics_by_panel_regime.csv'),
        provenance=bundle['provenance']+[train_folder/'manifest.json',label_folder/'manifest.json',
            grid_train_folder/'manifest.json',grid_eval_folder/'manifest.json',hybrid/'combined_predictions.parquet',
            hybrid/'metrics_by_panel_regime.csv',hybrid/'COMPLETE.json',grin/'GRIN_predictions.parquet',grin/'environment.json'])
