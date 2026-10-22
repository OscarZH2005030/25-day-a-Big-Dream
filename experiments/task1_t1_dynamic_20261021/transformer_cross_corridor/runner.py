from datetime import datetime, timezone
from pathlib import Path
import gc
import shutil
import time
import numpy as np
import pandas as pd
import torch
from task1.cache import snapshot
from task1.data import KEY, TARGETS, save_json, sha256
from task1.transformer_priority import lightgbm
from task1.transformer_priority.common import aligned, device_for, metric_rows, read, upstream, versions
from task1.transformer_priority.model import model_for
from task1.transformer_priority.runner import context
from task1.transformer_priority.train import checkpoint_read
from transformer_training.sampling import Sampler
from transformer_training.train import fit, predictions, verify_model, write_pointer
from transformer_training.runner import write_scores
from transformer_fusion.core import summarize
from transformer_fusion.runner import paired_bootstrap
from transformer_fusion.source import verify
from .common import (PANELS, VARIANTS, CONFIG, AUGMENT, EXPERIMENT, dependencies,
                     portable_dependencies, plan, methods, recipes, validate_config)
from .source import Prepared


class OriginalSampler:
    """Unmodified T0 sampling/inputs, with the existing exposure recorder adapter."""
    def __init__(self, data, mapping):
        self.data, self.mapping = data, mapping

    def batch(self, rng, aux_rng, config):
        batch, graph, spec = self.data.batch(rng, config, False)
        batch['original_target'] = batch['target'].clone()
        infos = []
        for i, (date, _) in enumerate(spec['samples']):
            n = int(batch['target'][i].sum())
            infos.append(dict(date=date, regime=self.mapping[date], pattern='none', original_targets=n,
                              retained_targets=n, extra_targets=0, supervised_targets=n))
        return batch, graph, spec, infos


def training_names(mode):
    return VARIANTS + (['T0'] if mode == 'rebuild_t0' else [])


def inputs(data, raw, tree, mode, role):
    dates = data.roles[role]
    ctx = context(data, dates)
    frames = {m:aligned(ctx[KEY], f)[TARGETS].to_numpy(np.float64) for m,f in raw.items()}
    baseline = dict(context=ctx, truth=data.truth(dates, role), LightGBM=aligned(ctx[KEY],tree)[TARGETS].to_numpy(np.float64),
                    T0=frames.get('T0',frames['T1']), T1=frames['T1'])
    result = dict(LightGBM=(baseline,dict(beta=0.,speed=1.,flow=1.)), T1=(baseline,dict(beta=1.)),
                  Current_fusion=(baseline,recipes(mode)))
    if mode == 'rebuild_t0':
        result['T0'] = (baseline,dict(beta=0.))
    candidate = dict(baseline,T1=frames['A_dynamic'])
    result['A_dynamic'] = (candidate,dict(beta=1.))
    result['A_dynamic_fixed'] = (candidate,recipes(mode))
    return result


def replay_internal(data, raw, model_records):
    rows = []
    truth = data.truth(data.roles['internal'],'internal')
    ctx = context(data,data.roles['internal'])
    for name, frame in raw.items():
        frame = aligned(ctx[KEY],frame)
        values = frame[TARGETS].to_numpy(np.float64,copy=True)
        values[:,1] /= ctx.lanes.to_numpy()
        score = np.mean([r['S_state'] for r in metric_rows(truth,values,ctx.mask_regime.to_numpy(),name,data.panel,data.family)])
        expected = model_records[name]['model']['selection']['best_internal_S_state']
        if abs(score-expected)>1e-10:
            raise ValueError('Original raw internal checkpoint score differs')
        rows.append(dict(panel=data.panel,model=name,replayed_S=score,checkpoint_S=expected,max_difference=abs(score-expected)))
    return rows


def validate_formal_scope(out, cfg):
    validate_config(cfg)
    checks = read(out/'integrity_check.json')
    for field in ['passed','all_models_frozen_before_holdout','isolated_exact','fixed_formula_exact','internal_checkpoint_replayed']:
        if not checks[field]:
            raise ValueError('Result invariant failed: '+field)
    if cfg['smoke']:
        return
    if cfg['training'] != CONFIG:
        raise ValueError('Formal budget differs')
    preflight = read(out/'preflight.json')
    refs = pd.read_csv(Path(cfg['project'])/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    detail = pd.read_csv(out/'holdout_metrics_by_panel_regime.csv')
    preservation = read(out/'holdout_preservation_checks.json')
    records = read(out/'model_cache_usage.json')
    for panel in PANELS:
        p = preflight['panels'][panel]
        for role, field in [('training','training_dates'),('internal','internal_validation_dates'),('holdout','holdout_dates')]:
            if p['dates'][role] != cfg['split'][field]:
                raise ValueError('Formal dates differ')
        for method in methods(cfg['fusion_mode']):
            g = detail.loc[detail.panel.eq(panel)&detail.method.eq(method)]
            item = preservation[panel+'/'+method]
            if set(g.regime) != {'R1','R2','R3'} or len(g)!=3 or g.n.sum()!=refs.loc[panel,'n']:
                raise ValueError('Formal score target scope differs')
            if item['isolated_rows']!=refs.loc[panel,'fallback_n'] or not item['isolated_exact']:
                raise ValueError('Original isolated fallback changed')
        for name in training_names(cfg['fusion_mode']):
            selection = records[panel][name]['model']['selection']
            if selection['config']!=CONFIG or selection['optimizer_updates']!=selection['epochs_run']*1024 or not 1<=selection['epochs_run']<=60:
                raise ValueError('Formal update budget differs')
    if read(out/'frozen_models.json')['selection_dates'] != cfg['split']['internal_validation_dates']:
        raise ValueError('Internal selection scope differs')


def verify_complete(out):
    complete = read(out/'COMPLETE.json')
    if complete['experiment']!=EXPERIMENT or not complete['integrity_passed']:
        raise ValueError('Experiment incomplete')
    if sha256(out/'artifact_manifest.json') != complete['artifact_manifest_sha256']:
        raise ValueError('Result manifest changed')
    verify(read(out/'artifact_manifest.json'))
    cfg = read(out/'run_config.json')
    validate_config(cfg)
    verify(cfg['implementation'])
    verify(read(out/'source_manifest.json'))
    frozen = read(out/'frozen_models.json')
    verify(frozen['internal_artifacts'])
    records = read(out/'model_cache_usage.json')
    for panel in PANELS:
        if set(records[panel]) != set(training_names(cfg['fusion_mode']))|{'LightGBM'}:
            raise ValueError('Missing/extra trained models')
        for name in training_names(cfg['fusion_mode']):
            record = records[panel][name]['model']
            verify_model(Path(record['folder']))
            if read(Path(record['folder'])/'best.json')!=record['best'] or frozen['models'][panel][name]!=record['best']:
                raise ValueError('Selected model changed')
    table = pd.read_csv(out/'holdout_comparison_methods.csv')
    if set(table.method)!=set(methods(cfg['fusion_mode'])) or table.method.duplicated().any():
        raise ValueError('Comparison scope differs')
    rebuilt = summarize(pd.read_csv(out/'holdout_metrics_by_panel_regime.csv')).set_index('method')
    for row in table.itertuples(index=False):
        for field in ['S_state','rmse_speed_kmh','rmse_flow_per_lane']:
            if abs(getattr(row,field)-rebuilt.loc[row.method,field])>1e-10:
                raise ValueError('Score aggregation differs')
        for field,reference in [('delta_vs_T1','T1'),('delta_vs_current_fusion','Current_fusion')]:
            if abs(getattr(row,field)-(row.S_state-rebuilt.loc[reference,'S_state']))>1e-12:
                raise ValueError('Score delta differs')
    validate_formal_scope(out,cfg)
    return cfg,table


def log_results(out,project):
    cfg,table = verify_complete(out)
    if cfg['smoke']:
        result = dict(status='skipped_smoke')
    else:
        log = project/'实验日志.txt'
        marker = f'实验编号：{out.name}；结果目录：{out}'
        with upstream().lock(project/'.d12_i5_dynamic_log.lock'):
            old = log.read_text(encoding='utf-8-sig') if log.exists() else ''
            if marker in old:
                result = dict(status='already_logged',log_path=str(log))
            else:
                formula = '45%T0+45%候选+10%LightGBM' if cfg['fusion_mode']=='rebuild_t0' else '90%候选+10%LightGBM'
                lines = ['', '【D12_I5跨走廊T1动态缺失复现｜已完成】',marker,
                    f'四个主模型：D12_I5_N/S各T1、A_dynamic；融合模式={cfg["fusion_mode"]}，公式={formula}。',
                    '28日历史/168日训练/22日内部/55日留出；原51特征、初始化、60×1024上限和早停；实际轮数见selection.json。',
                    '共享LightGBM按原配置准备，所有神经模型及内部结果冻结后才读留出；两通道孤立节点精确回退。',
                    'T0如补建仅作本机同口径基准，不将历史十路汇总冒充本机模型；不调整固定融合权重。']
                for row in table.itertuples(index=False):
                    lines.append(f'{row.method}：S={row.S_state:.9f}；对T1 ΔS={row.delta_vs_T1:+.9f}；对固定对照 ΔS={row.delta_vs_current_fusion:+.9f}；速度宏RMSE={row.rmse_speed_kmh:.6f}；每车道流量宏RMSE={row.rmse_flow_per_lane:.6f}。')
                lines += ['逐Regime/缺口/日期/拥堵/孤立、曝光/覆盖、描述性配对区间与来源缓存记录见结果目录。',
                          '本轮仅两个方向一个新走廊家族、单种子本地比较；不宣称十路突破0.936910。','']
                with log.open('a',encoding='utf-8',newline='') as stream:
                    stream.write('\n'.join(lines)+'\n')
                result = dict(status='appended',log_path=str(log))
    save_json(out/'log_update.json',result)
    return result


def run(args):
    started = time.perf_counter()
    out = args.output
    if out==args.cache_dir or out.is_relative_to(args.cache_dir) or args.cache_dir.is_relative_to(out):
        raise ValueError('Output and cache must be separate')
    if out.is_relative_to(args.release_root) or args.cache_dir.is_relative_to(args.release_root):
        raise ValueError('Outputs cannot be written into released inputs')
    config = dict(CONFIG)
    if args.smoke:
        config.update(epochs=2,steps_per_epoch=4,patience=2)
    cfg = dict(experiment=EXPERIMENT,project=str(args.project),release_root=str(args.release_root),
        cache_dir=str(args.cache_dir),device=args.device,smoke=args.smoke,action=args.action,
        panels=PANELS,variants=VARIANTS,training=config,augmentation=AUGMENT,split=plan(),
        fusion_mode=args.fusion_mode,methods=methods(args.fusion_mode),fixed_fusion=recipes(args.fusion_mode),
        implementation=dependencies(),versions=versions())
    validate_config(cfg)
    device = device_for(args.device)
    torch.set_num_threads(config['threads'])
    out.mkdir(parents=True,exist_ok=True)
    with upstream().lock(out/'execution.lock'):
        if (out/'run_config.json').exists():
            if not args.resume or read(out/'run_config.json')!=cfg:
                raise ValueError('Existing output differs; unchanged --resume or new output required')
            if (out/'COMPLETE.json').exists():
                print(log_results(out,args.project),flush=True)
                return
        save_json(out/'run_config.json',cfg)
        def status(**values):
            write_pointer(out/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**values))
        try:
            source = Prepared(args,cfg['split'])
            mappings = {}
            for panel in PANELS:
                status(stage='preparing_features',panel=panel)
                _,mappings[panel] = source.prepare(panel)
            save_json(out/'preflight.json',dict(passed=True,panels=source.preflight,holdout_labels_read=False,formal_training_started=False))
            save_json(out/'feature_parity.json',source.parity)
            if args.action=='prepare':
                source.finish(out)
                status(stage='prepared',formal_neural_training_started=False,holdout_labels_read=False)
                print('Full feature/observation preflight passed; no formal model training or holdout labels.',flush=True)
                return
            if device.type=='cuda':
                torch.cuda.reset_peak_memory_stats(device)
            records = {}
            internal_inputs = {}
            replay = []
            for panel in PANELS:
                data = source.data[panel]
                records[panel] = {}
                status(stage='shared_lightgbm',panel=panel)
                trees,tree_record = lightgbm.fit(data,args.smoke,config['threads'])
                tree_frame,pred_record = lightgbm.predictions(data,trees,tree_record,data.roles['internal'],config['threads'])
                records[panel]['LightGBM'] = dict(model=tree_record,internal_predictions=pred_record)
                del trees
                raw = {}
                _,initial = model_for(config,'T1_balanced')
                source_identity = dict(expected_initial_sha256=initial,implementation=portable_dependencies(),
                                       versions=cfg['versions'],device=str(device),experiment=EXPERIMENT)
                for name in training_names(args.fusion_mode):
                    status(stage='training_start',panel=panel,variant=name)
                    sampler = (source.samplers[panel] if name=='A_dynamic' else Sampler(data,name,mappings[panel])) if name!='T0' else OriginalSampler(data,mappings[panel])
                    model,record = fit(data,sampler,config,name,device,args.cache_dir,source_identity,status)
                    folder = out/'models'/panel/name
                    folder.mkdir(parents=True,exist_ok=True)
                    for filename in ['selection.json','training_history.csv','regime_exposure.csv']:
                        shutil.copyfile(Path(record['folder'])/filename,folder/filename)
                    raw[name],prediction_record = predictions(data,model,record,data.roles['internal'],config,device,'internal')
                    raw[name].to_parquet(folder/'internal_raw_predictions.parquet',index=False)
                    records[panel][name] = dict(model=record,internal_predictions=prediction_record)
                    save_json(out/'partial_model_cache_usage.json',records)
                    del model
                    gc.collect()
                    if device.type=='cuda':
                        torch.cuda.empty_cache()
                replay.extend(replay_internal(data,raw,records[panel]))
                internal_inputs[panel] = inputs(data,raw,tree_frame,args.fusion_mode,'internal')
            write_scores(internal_inputs,out,'internal')
            pd.DataFrame(replay).to_csv(out/'internal_checkpoint_replay.csv',index=False)
            frozen = dict(selection_dates=cfg['split']['internal_validation_dates'],
                models={p:{m:records[p][m]['model']['best'] for m in training_names(args.fusion_mode)} for p in PANELS},
                trees={p:records[p]['LightGBM']['model']['key'] for p in PANELS},
                fixed_fusion=cfg['fixed_fusion'],frozen_utc=datetime.now(timezone.utc).isoformat(),
                internal_artifacts=snapshot([out/'internal_comparison_methods.csv',out/'internal_metrics_by_panel_regime.csv',
                    out/'internal_checkpoint_replay.csv']+[out/'models'/p/m/'selection.json' for p in PANELS for m in training_names(args.fusion_mode)]))
            save_json(out/'frozen_models.json',frozen)
            frozen_hash = sha256(out/'frozen_models.json')
            del internal_inputs
            gc.collect()
            status(stage='all_models_frozen_before_holdout')
            holdout_inputs = {}
            for panel in PANELS:
                if sha256(out/'frozen_models.json')!=frozen_hash:
                    raise ValueError('Frozen models changed')
                verify(frozen['internal_artifacts'])
                data = source.data[panel]
                data.allow_holdout = True
                tree_record = records[panel]['LightGBM']['model']
                source.cache_manifest(tree_record['folder'])
                trees = [lightgbm.read_booster(Path(tree_record['folder'])/(n+'.txt')) for n in ['speed_kmh','flow_per_lane']]
                tree_frame,pred_record = lightgbm.predictions(data,trees,tree_record,data.roles['holdout'],config['threads'])
                records[panel]['LightGBM']['holdout_predictions'] = pred_record
                del trees
                raw = {}
                for name in training_names(args.fusion_mode):
                    status(stage='holdout_inference',panel=panel,variant=name)
                    record = records[panel][name]['model']
                    verify_model(Path(record['folder']))
                    model,_ = model_for(config,'T1_balanced')
                    checkpoint,_ = checkpoint_read(Path(record['folder']),'best.json',record['key'])
                    model.load_state_dict(checkpoint['state_dict'])
                    model = model.to(device)
                    raw[name],pred_record = predictions(data,model,record,data.roles['holdout'],config,device,'holdout')
                    records[panel][name]['holdout_predictions'] = pred_record
                    raw[name].to_parquet(out/'models'/panel/name/'holdout_raw_predictions.parquet',index=False)
                    del model,checkpoint
                    gc.collect()
                    if device.type=='cuda':
                        torch.cuda.empty_cache()
                holdout_inputs[panel] = inputs(data,raw,tree_frame,args.fusion_mode,'holdout')
            _,table,daily,preservation = write_scores(holdout_inputs,out,'holdout')
            intervals = []
            for reference,candidate in [('T1','A_dynamic'),('Current_fusion','A_dynamic_fixed')]:
                pair = daily.loc[daily.method.isin([reference,candidate])].copy()
                pair.loc[pair.method.eq(reference),'method'] = 'T0'
                for row in paired_bootstrap(pair,[candidate]):
                    row['reference'] = reference
                    intervals.append(row)
            pd.DataFrame(intervals).to_csv(out/'paired_calendar_bootstrap.csv',index=False)
            if sha256(out/'frozen_models.json')!=frozen_hash:
                raise ValueError('Frozen model manifest changed')
            save_json(out/'model_cache_usage.json',records)
            source.finish(out)
            verify(cfg['implementation'])
            save_json(out/'integrity_check.json',dict(passed=True,all_models_frozen_before_holdout=True,
                isolated_exact=all(x['isolated_exact'] for x in preservation.values()),fixed_formula_exact=True,
                internal_checkpoint_replayed=True,full_dates=not args.smoke,main_neural_models=4,
                auxiliary_t0_models=2 if args.fusion_mode=='rebuild_t0' else 0,
                limitation='One new two-direction family; local exploratory evidence, not a ten-direction score'))
            save_json(out/'memory.json',dict(device=str(device),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None))
            manifest = snapshot([p for p in out.rglob('*') if p.is_file() and p.name not in ['execution.lock','status.json','COMPLETE.json','artifact_manifest.json','log_update.json']])
            save_json(out/'artifact_manifest.json',manifest)
            save_json(out/'COMPLETE.json',dict(experiment=EXPERIMENT,smoke=args.smoke,integrity_passed=True,
                artifact_manifest_sha256=sha256(out/'artifact_manifest.json'),seconds=time.perf_counter()-started))
            result = log_results(out,args.project)
            status(stage='completed',log_status=result['status'],seconds=time.perf_counter()-started)
            print(table.to_string(index=False),flush=True)
            print('Completed:',out,'Log:',result['status'],flush=True)
        except Exception as error:
            status(stage='failed',error=repr(error))
            raise
