from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..cache import feature_implementation
from ..cache_builder import IDENTIFIER_COLUMNS
from ..data import KEY,TARGETS,sha256
from ..fusion_experiments import aligned_prediction
from ..grin.hybrid import load_baseline,require_compatible_grin,switch_predictions
from ..pipeline import model_parameters
from ..target_labels import aligned_direct_targets
from ..traffic_wave.reference import verified_manifest
from ..traffic_wave.core import DEFAULTS
from ..speed_fusion.reference import load_context
from ..speed_fusion.rules import CONTROL,NEURAL,WEIGHTS,PRIMARY,METHODS

PROJECT=Path(__file__).resolve().parents[2]
PANEL='D7_I405_N'
TRAIN_ROWS=214000
EVAL_ROWS=819695
OLD_METHOD='Baseline_GRIN_gap60'
SPLIT=dict(fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',eval_end='2031-01-31')
NN_FIELDS=['window','stride','hidden','layers','epochs','steps_per_epoch','patience','learning_rate','seed','batch_size','threads']
NN_EXPECTED=dict(window=48,stride=24,hidden=32,layers=6,epochs=30,steps_per_epoch=100,patience=5,learning_rate=.001,seed=20261016,batch_size=2,threads=4)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def inside(path):
    result=Path(path).resolve()
    if not result.is_relative_to(PROJECT):raise ValueError('Path escapes project: '+str(result))
    return result


def verify_files(items):
    for item in items:
        path=inside(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Frozen input/implementation changed: '+str(path))


def check_model_pair(pair,training,label_manifest,params):
    identity=pair['identity']
    if identity['training_key']!=training['key'] or identity['direct_labels_key']!=label_manifest['key'] or identity['parameters']!=params or identity['rounds']!=350:
        raise ValueError('D7 original model-pair metadata differs')


def validate_recipe(wave,neural,fusion,threads=4,batch_size=2):
    if any(wave.get(k)!=v for k,v in DEFAULTS.items()) or wave.get('rounds')!=350:
        raise ValueError('I5 fixed traffic-wave recipe differs')
    if any(neural.get(k)!=v for k,v in NN_EXPECTED.items()):raise ValueError('I5 fixed neural recipe differs')
    if fusion.get('weights')!=WEIGHTS or fusion.get('primary_method')!=PRIMARY:
        raise ValueError('I5 fixed fusion weights or primary differ')
    if threads!=neural['threads'] or batch_size!=neural['batch_size']:
        raise ValueError('Controlled D7 validation fixes I5 threads=4 and batch-size=2')
    for cfg in [wave,neural,fusion]:
        if cfg.get('panel')!='D12_I5_N' or cfg.get('smoke') or any(cfg.get(k)!=v for k,v in SPLIT.items()):
            raise ValueError('I5 formal panel/split differs')


def frozen_recipe(args):
    runs=dict(wave=inside(args.i5_wave_run),neural=inside(args.i5_neural_run),fusion=inside(args.i5_fusion_run))
    configs={name:read(folder/'run_config.json') for name,folder in runs.items()}
    expected_experiments=dict(wave='traffic_wave',neural='wave_context',fusion='speed_fusion')
    provenance=[]
    for name,folder in runs.items():
        complete=read(folder/'COMPLETE.json')
        if complete.get('experiment')!=expected_experiments[name] or complete.get('smoke') or not all(complete.get(k) for k in ['reference_passed','pairing_passed','integrity_passed']):
            raise ValueError('I5 template must be completed formal: '+name)
        verify_files(read(folder/'environment.json')['source_files'])
        provenance += [folder/f for f in ['run_config.json','COMPLETE.json','environment.json','metrics_by_panel_regime.csv']]
    manifest=runs['fusion']/'artifact_manifest.json'
    if sha256(manifest)!=read(runs['fusion']/'COMPLETE.json')['artifact_manifest_sha256']:raise ValueError('I5 fusion manifest changed')
    verify_files(read(manifest)['artifacts'])
    validate_recipe(configs['wave'],configs['neural'],configs['fusion'],args.threads,args.batch_size)
    if inside(configs['fusion']['source_run'])!=runs['neural'] or inside(configs['neural']['wave_run'])!=runs['wave']:
        raise ValueError('I5 template lineage differs')
    return dict(wave_parameters={k:configs['wave'][k] for k in DEFAULTS},wave_rounds=350,
        neural={k:configs['neural'][k] for k in NN_FIELDS},weights=WEIGHTS,primary_method=PRIMARY,
        sources={k:str(v) for k,v in runs.items()},provenance=provenance+[manifest],
        i5_comparison=pd.read_csv(runs['fusion']/'comparison_methods.csv'))


def load_sources(args,recipe):
    bundle=load_baseline(inside(args.baseline_run),PANEL);bc=bundle['config']
    if any(bc.get(k)!=v for k,v in SPLIT.items()) or bc.get('sample_per_day')!=1000 or bc.get('seed')!=20261016:
        raise ValueError('D7 original target protocol differs')
    root=inside(bc['release_root']);cache=inside(bc['cache_dir'])
    records=read(inside(args.baseline_run)/'cache_usage.json');record=next(r for r in records if r['panel']==PANEL)
    train_folder=bundle['training_table'].parent;training=verified_manifest(train_folder)
    if training['identity']['implementation']!=feature_implementation():raise ValueError('Original feature implementation differs')
    eval_folder=cache/record['evaluation_key'];evaluation=verified_manifest(eval_folder)
    verify_files(evaluation['identity']['inputs'])
    label_folder=cache/record['direct_labels_key'];label_manifest=verified_manifest(label_folder)
    if label_manifest['identity']['parent_training_key']!=training['key'] or label_manifest['identity']['builder_sha256']!=sha256(PROJECT/'task1/target_labels.py'):
        raise ValueError('D7 direct-label parent/implementation differs')
    # D7's existing model-pair cache replaces the I5 standalone models/ directory layout.
    pair_folder=cache/record['model_pair_key'];pair=verified_manifest(pair_folder)
    params=model_parameters(dict(seed=20261016,threads=args.threads))
    check_model_pair(pair,training,label_manifest,params)
    train=pd.read_parquet(bundle['training_table']);keys=train[IDENTIFIER_COLUMNS].rename(columns={k:k[2:] for k in IDENTIFIER_COLUMNS}).reset_index(drop=True)
    labels=aligned_direct_targets(train,pd.read_parquet(label_folder/'direct_targets.parquet'))
    hybrid=inside(args.hybrid_run);hc,hcomplete=read(hybrid/'run_config.json'),read(hybrid/'COMPLETE.json')
    if hcomplete.get('experiment')!='grin_hybrid' or hc.get('smoke') or not hcomplete.get('reference_passed') or hc['panel']!=PANEL or inside(hc['baseline_run'])!=inside(args.baseline_run):
        raise ValueError('D7 old hybrid source differs')
    grin=inside(hcomplete['grin_source']);gc=read(grin/'run_config.json');require_compatible_grin(grin,gc)
    if gc['panel']!=PANEL or gc['seed']!=20261016 or gc['sample_per_day']!=1000 or any(gc[k]!=v for k,v in SPLIT.items()):
        raise ValueError('D7 GRIN protocol differs')
    gu=read(grin/'cache_usage.json');gcache=inside(gc['cache_dir'])
    gt_folder,ge_folder=gcache/gu['training_key'],gcache/gu['evaluation_key']
    gt,ge=verified_manifest(gt_folder),verified_manifest(ge_folder)
    if not keys.equals(pd.read_parquet(gt_folder/'training_sample_keys.parquet')[KEY].reset_index(drop=True)) or len(keys)!=TRAIN_ROWS:
        raise ValueError('D7 baseline/GRIN ordered training keys differ')
    expected=pd.read_parquet(ge_folder/'expected_keys.parquet')
    if len(expected)!=EVAL_ROWS or len(gt['metadata']['nodes'])!=352 or gt['metadata']['nodes']!=ge['metadata']['nodes']:
        raise ValueError('D7 grid/target/node scope differs')
    fit_dates=pd.date_range(SPLIT['fit_start'],SPLIT['fit_end']).strftime('%Y-%m-%d').tolist()
    eval_dates=pd.date_range(SPLIT['eval_start'],SPLIT['eval_end']).strftime('%Y-%m-%d').tolist()
    if gt['metadata']['dates']!=fit_dates or ge['metadata']['dates']!=eval_dates or training['metadata']['fit_dates']!=fit_dates or evaluation['metadata']['eval_dates']!=eval_dates:
        raise ValueError('D7 full dates differ')
    diagnostics=load_context(eval_folder,evaluation)
    diagnostics=expected[KEY].merge(diagnostics,on=KEY,validate='one_to_one',sort=False)
    if len(diagnostics)!=EVAL_ROWS:raise ValueError('D7 observable context scope differs')
    baseline=aligned_prediction(expected,bundle['baseline'])
    grin_pred=aligned_prediction(expected,pd.read_parquet(grin/'GRIN_predictions.parquet'))
    current=aligned_prediction(expected,pd.read_parquet(hybrid/'combined_predictions.parquet'))
    rebuilt=switch_predictions(expected,baseline,grin_pred,diagnostics.gap_minutes)
    if not np.array_equal(current[TARGETS].to_numpy(),rebuilt[TARGETS].to_numpy()):raise ValueError('D7 old combination does not reconstruct exactly')
    provenance=bundle['provenance']+recipe['provenance']+[train_folder/'manifest.json',label_folder/'manifest.json',pair_folder/'manifest.json',
        gt_folder/'manifest.json',ge_folder/'manifest.json',hybrid/'COMPLETE.json',hybrid/'run_config.json',hybrid/'combined_predictions.parquet',
        hybrid/'metrics_by_panel_regime.csv',grin/'GRIN_predictions.parquet',grin/'environment.json']
    return dict(bundle=bundle,root=root,train=train,keys=keys,labels=labels,training=training,evaluation=evaluation,
        eval_folder=eval_folder,grid_train_folder=gt_folder,grid_eval_folder=ge_folder,grid_train=gt,grid_eval=ge,
        expected=expected,diagnostics=diagnostics,current=current,baseline=baseline,grin=grin_pred,hybrid_path=hybrid,
        reference=pd.read_csv(hybrid/'metrics_by_panel_regime.csv'),provenance=provenance)


def control_reference(source,method,dates,smoke):
    if not smoke:return source['reference'].loc[source['reference'].method.eq(method)]
    d=pd.read_csv(source['hybrid_path']/(method+'_daily_errors.csv'));d=d.loc[d.date.isin(dates)]
    rows=[]
    for regime,g in d.groupby('mask_regime'):
        v=np.sqrt(np.average(g.speed_mse,weights=g.n));q=np.sqrt(np.average(g.flow_lane_mse,weights=g.n))
        rows.append(dict(method=method,panel=PANEL,regime=regime,n=int(g.n.sum()),rmse_speed_kmh=v,rmse_flow_per_lane=q,
            S_state=.54*max(0,1-v/25)+.46*max(0,1-q/600)))
    return pd.DataFrame(rows)
