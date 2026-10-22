from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import gc
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
from task1.cache import snapshot
from task1.data import KEY,TARGETS,save_json,sha256
from task1.transformer_priority.common import PILOT,read,upstream,aligned,versions
from task1.transformer_priority.model import model_for
from task1.transformer_priority.train import checkpoint_read
from transformer_fusion.core import details,evaluate,summarize
from transformer_fusion.runner import paired_bootstrap,verify_complete as verify_fusion
from transformer_fusion.source import verify
from .common import CONFIG,AUGMENT,VARIANTS,METHODS,EXPERIMENT,implementation
from .source import TrainingSource
from .sampling import Sampler
from .train import fit,predictions,verify_model

RECIPES={'T0':dict(beta=0.),'T1':dict(beta=1.),'Current_equal':dict(beta=.5),
         'Current_fusion':dict(beta=.5,speed=.1,flow=.1)}


def method_inputs(baseline,new_predictions):
    result={name:(baseline,recipe) for name,recipe in RECIPES.items()}
    for variant,frame in new_predictions.items():
        bundle=dict(baseline,T1=frame[TARGETS].to_numpy(np.float64))
        result[variant]=(bundle,dict(beta=1.))
        result[variant+'_fixed']=(bundle,dict(beta=.5,speed=.1,flow=.1))
    return result


def write_scores(inputs,out,phase):
    rows=[];daily=[];preservation={}
    for panel,methods in inputs.items():
        dest=out/phase/panel;dest.mkdir(parents=True,exist_ok=True)
        for method,(bundle,recipe) in methods.items():
            rows.append(details({panel:bundle},recipe,method))
            values,groups=evaluate(bundle,recipe,method)
            ctx=bundle['context'];iso=ctx.isolated.to_numpy(bool)
            neural=bundle['T0'] if recipe['beta']==0 else bundle['T1'] if recipe['beta']==1 else .5*bundle['T0']+.5*bundle['T1']
            expected=neural.copy()
            for channel,field in enumerate(['speed','flow']):
                alpha=recipe.get(field,0.)
                if alpha: expected[:,channel]=(1-alpha)*neural[:,channel]+alpha*bundle['LightGBM'][:,channel]
            expected[iso]=bundle['LightGBM'][iso]
            if not np.array_equal(values,expected): raise ValueError('Fixed replacement formula differs')
            exact=np.array_equal(values[iso],bundle['LightGBM'][iso])
            if not exact: raise ValueError('Isolated fallback changed')
            preservation[panel+'/'+method]=dict(rows=len(values),isolated_rows=int(iso.sum()),isolated_exact=True)
            frame=ctx[KEY].copy();frame[TARGETS]=values
            frame.to_parquet(dest/(method+'_predictions.parquet'),index=False)
            for name,group in groups.items(): group.to_csv(dest/(method+'_'+name+'_errors.csv'),index=False)
            daily.append(groups['daily'])
    detail=pd.concat(rows,ignore_index=True);table=summarize(detail)
    reference=table.set_index('method').S_state
    table['delta_vs_T1']=table.S_state-reference['T1']
    table['delta_vs_current_fusion']=table.S_state-reference['Current_fusion']
    detail.to_csv(out/(phase+'_metrics_by_panel_regime.csv'),index=False)
    table.to_csv(out/(phase+'_comparison_methods.csv'),index=False)
    daily=pd.concat(daily,ignore_index=True);daily.to_csv(out/(phase+'_daily_errors.csv'),index=False)
    save_json(out/(phase+'_preservation_checks.json'),preservation)
    return detail,table,daily,preservation


def verify_complete(out):
    complete=read(out/'COMPLETE.json')
    if complete['experiment']!=EXPERIMENT or not complete['integrity_passed']: raise ValueError('Experiment incomplete')
    if sha256(out/'artifact_manifest.json')!=complete['artifact_manifest_sha256']: raise ValueError('Result manifest changed')
    artifacts=read(out/'artifact_manifest.json');verify(artifacts)
    names={str(Path(x['path']).relative_to(out)) for x in artifacts}
    required={'run_config.json','frozen_models.json','source_manifest.json','source_cache_usage.json',
              'model_cache_usage.json','integrity_check.json','holdout_comparison_methods.csv',
              'holdout_metrics_by_panel_regime.csv','holdout_preservation_checks.json','control_score_replay.csv'}
    required.update(str(Path('holdout')/panel/(method+'_predictions.parquet')) for panel in PILOT for method in METHODS)
    if not required.issubset(names): raise ValueError('Missing result artifacts')
    cfg=read(out/'run_config.json')
    verify(cfg['implementation']);verify(read(out/'source_manifest.json'))
    frozen=read(out/'frozen_models.json');verify(frozen['internal_artifacts'])
    records=read(out/'model_cache_usage.json')
    for panel in PILOT:
        for variant in VARIANTS:
            record=records[panel][variant]['model'];verify_model(Path(record['folder']))
            if read(Path(record['folder'])/'best.json')!=record['best']: raise ValueError('Selected model changed')
    detail=pd.read_csv(out/'holdout_metrics_by_panel_regime.csv')
    table=pd.read_csv(out/'holdout_comparison_methods.csv')
    if set(table.method)!=set(METHODS) or table.method.duplicated().any(): raise ValueError('Missing method')
    rebuilt=summarize(detail).set_index('method')
    for row in table.itertuples(index=False):
        for field in ['S_state','rmse_speed_kmh','rmse_flow_per_lane']:
            if abs(getattr(row,field)-rebuilt.loc[row.method,field])>1e-10: raise ValueError('Score aggregation differs')
        if abs(row.delta_vs_T1-(row.S_state-rebuilt.loc['T1','S_state']))>1e-12: raise ValueError('T1 delta differs')
        if abs(row.delta_vs_current_fusion-(row.S_state-rebuilt.loc['Current_fusion','S_state']))>1e-12: raise ValueError('Fusion delta differs')
    return cfg,table


def log_results(out,project):
    cfg,table=verify_complete(out)
    if cfg['smoke']:
        result=dict(status='skipped_smoke');save_json(out/'log_update.json',result);return result
    if cfg['training']!=CONFIG or cfg['panels']!=PILOT or cfg['variants']!=VARIANTS or cfg['augmentation']!=AUGMENT:
        raise ValueError('Formal configuration differs')
    frozen=read(out/'frozen_models.json')
    if frozen['selection_dates']!=cfg['split']['internal_validation_dates']:
        raise ValueError('Formal internal date scope differs')
    checks=read(out/'integrity_check.json')
    if not all(checks[k] for k in ['passed','all_four_models_frozen_before_holdout','control_scores_replayed','isolated_exact','fixed_formula_exact']):
        raise ValueError('Formal invariant failed')
    detail=pd.read_csv(out/'holdout_metrics_by_panel_regime.csv')
    expected=pd.read_csv(project/'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    preservation=read(out/'holdout_preservation_checks.json')
    for panel in PILOT:
        for method in METHODS:
            g=detail.loc[detail.panel.eq(panel)&detail.method.eq(method)]
            p=preservation[panel+'/'+method]
            if len(g)!=3 or set(g.regime)!={'R1','R2','R3'} or g.n.sum()!=expected.loc[panel,'n'] or p['isolated_rows']!=expected.loc[panel,'fallback_n'] or not p['isolated_exact']:
                raise ValueError('Formal target scope differs')
    model_records=read(out/'model_cache_usage.json')
    for panel in PILOT:
        for variant in VARIANTS:
            selection=model_records[panel][variant]['model']['selection']
            if selection['config']!=CONFIG or selection['optimizer_updates']>60*1024:
                raise ValueError('Training budget differs')
    log=project/'实验日志.txt';marker=f'实验编号：{out.name}；结果目录：{out}'
    with upstream().lock(project/'.transformer_training_log.lock'):
        previous=log.read_text(encoding='utf-8-sig') if log.exists() else ''
        if marker in previous: result=dict(status='already_logged',log_path=str(log))
        else:
            lines=['',f'【两路T1动态缺失/Regime均衡训练消融｜{datetime.now(ZoneInfo("America/Los_Angeles")).date()}｜已完成】',marker,
                '原历史28/训练168/内部22/留出55日；仅A_dynamic、B_regime两方向四个新模型；T0/T1/LightGBM及原固定融合复用。',
                '原T1相同初始化、结构、60轮×1024步上限、batch2、原AdamW/调度/早停；A最多替换20%监督，遮挡后重算51特征；B每步等概率选R再抽两窗口，不叠类别权重。',
                '新模型单独输出与45%T0+45%新模型+10%LightGBM固定融合均按原孤立回退；四个模型与内部产物冻结后才读取留出标签。',
                '当前仅两方向一走廊家族，55日已反复检查，属探索性本地比较，不能据此宣称十路超过0.936910。']
            for row in table.itertuples(index=False):
                lines.append(f'{row.method}：S={row.S_state:.9f}；对T1 ΔS={row.delta_vs_T1:+.9f}；对原固定融合 ΔS={row.delta_vs_current_fusion:+.9f}；速度宏RMSE={row.rmse_speed_kmh:.6f}；每车道流量宏RMSE={row.rmse_flow_per_lane:.6f}。')
            lines+=['四模型训练曲线、各R曝光/监督与损失、覆盖摘要、缓存/断点、逐regime/日期/缺口/拥堵、描述性区间及来源hash均见结果目录。','']
            with log.open('a',encoding='utf-8',newline='') as stream: stream.write('\n'.join(lines)+'\n')
            result=dict(status='appended',log_path=str(log))
    save_json(out/'log_update.json',result);return result


def run(args):
    started=time.perf_counter();out=args.output
    for protected in [args.source_run,args.fusion_run]:
        if out.is_relative_to(protected) or protected.is_relative_to(out) or args.cache_dir.is_relative_to(protected):
            raise ValueError('New output/cache must be separate from protected sources')
    config=dict(CONFIG)
    if args.smoke: config.update(epochs=2,steps_per_epoch=4,patience=2)
    out.mkdir(parents=True,exist_ok=True)
    cfg=dict(experiment=EXPERIMENT,source_run=str(args.source_run),fusion_run=str(args.fusion_run),
        project=str(args.project),cache_dir=str(args.cache_dir),device=args.device,smoke=args.smoke,action=args.action,
        panels=PILOT,variants=VARIANTS,methods=METHODS,training=config,augmentation=AUGMENT,
        fixed_fusion=dict(beta=.5,speed=.1,flow=.1),implementation=implementation(),versions=versions(),
        source_anchors=snapshot([args.source_run/'COMPLETE.json',args.source_run/'artifact_manifest.json',
                                args.fusion_run/'COMPLETE.json',args.fusion_run/'artifact_manifest.json']))
    with upstream().lock(out/'execution.lock'):
        if (out/'run_config.json').exists():
            old=read(out/'run_config.json');old.pop('split',None)
            if not args.resume or old!=cfg: raise ValueError('Existing output/config changed; use unchanged --resume or a new output')
            if (out/'COMPLETE.json').exists(): print(log_results(out,args.project),flush=True);return
        def status(**values):
            save_json(out/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**values))
        try:
            status(stage='verifying_frozen_sources')
            _,fusion_cfg,_=verify_fusion(args.fusion_run)
            if Path(fusion_cfg['source_run'])!=args.source_run or read(args.fusion_run/'selection.json')['recipe']!=cfg['fixed_fusion']:
                raise ValueError('Frozen fusion source differs')
            source=TrainingSource(args.source_run,args.project,args.cache_dir,out,args.smoke,args.device,fusion_run=args.fusion_run)
            source.add(snapshot([args.fusion_run/n for n in ['COMPLETE.json','artifact_manifest.json','selection.json','metrics_by_panel_regime.csv']]))
            cfg['split']=source.plan;save_json(out/'run_config.json',cfg)
            device=source.device;samplers={};preflight={};parity={}
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            for panel in PILOT:
                status(stage='preparing_original_training_cache',panel=panel);print('[prepare training]',panel,flush=True)
                data=source.prepare(panel)
                sampler=Sampler(data,'A_dynamic',source.date_regime[panel]);samplers[panel]=sampler
                # Recompute unmodified observations once and require bitwise original-feature parity.
                first=data.roles['training'][0]
                from .augment import features_from_observations
                rebuilt=features_from_observations(data.exp,first,sampler.observations.get(first),np.arange(data.n))
                for name,values in zip(['features','base','visible','reliability'],rebuilt[:4]):
                    np.testing.assert_array_equal(values,data.array(first,name))
                parity[panel]=dict(passed=True,date=first,all_51_features_exact=True)
                roles=source.roles[panel]
                source.saved_holdout(panel,data,roles['holdout'])
                preflight[panel]=dict(dates=roles,rows={r:sum(data.records[d]['rows'] for d in dates) for r,dates in roles.items()},
                    training_regimes={r:sum(v==r for v in source.date_regime[panel].values()) for r in ['R1','R2','R3']})
                if args.action=='prepare':
                    for date in roles['training']: sampler.observations.ensure(date)
            save_json(out/'preflight.json',dict(passed=True,panels=preflight,holdout_labels_read=False,formal_training_started=False))
            save_json(out/'feature_parity.json',parity)
            if args.action=='prepare':
                finish_source(source,out,samplers)
                status(stage='prepared',passed=True);print('Full date/cache preflight passed; no model training or holdout labels.',flush=True);return
            internal_base={};models={};internal_new={}
            for panel in PILOT:
                data=source.data[panel];models[panel]={};internal_new[panel]={}
                status(stage='reusing_internal_baselines',panel=panel)
                internal_base[panel]=source.bundle(panel,'internal')
                for variant in VARIANTS:
                    sampler=samplers[panel] if variant=='A_dynamic' else Sampler(data,variant,source.date_regime[panel])
                    expected=source.records[panel]['models']['T1_balanced']['model']['selection']['common_initial_sha256']
                    source_identity=dict(expected_initial_sha256=expected,source_anchors=cfg['source_anchors'],
                        original_implementation=source.cfg['implementation'],device=str(device))
                    status(stage='training_start',panel=panel,variant=variant)
                    model,record=fit(data,sampler,config,variant,device,args.cache_dir,source_identity,status)
                    folder=out/'models'/panel/variant;folder.mkdir(parents=True,exist_ok=True)
                    for name in ['selection.json','training_history.csv','regime_exposure.csv']:
                        shutil.copyfile(Path(record['folder'])/name,folder/name)
                    raw,pred_record=predictions(data,model,record,data.roles['internal'],config,device,'internal')
                    raw=aligned(data.expected(data.roles['internal']),raw)
                    raw.to_parquet(folder/'internal_raw_predictions.parquet',index=False)
                    internal_new[panel][variant]=raw
                    models[panel][variant]=dict(model=record,internal_predictions=pred_record)
                    save_json(out/'partial_model_cache_usage.json',models)
                    del model;gc.collect()
                    if device.type=='cuda': torch.cuda.empty_cache()
            internal_inputs={p:method_inputs(internal_base[p],internal_new[p]) for p in PILOT}
            write_scores(internal_inputs,out,'internal')
            frozen=dict(selection_dates=source.internal_dates(),models={p:{v:models[p][v]['model']['best'] for v in VARIANTS} for p in PILOT},
                        fixed_fusion=cfg['fixed_fusion'],frozen_utc=datetime.now(timezone.utc).isoformat(),
                        internal_artifacts=snapshot([out/'internal_comparison_methods.csv',out/'internal_metrics_by_panel_regime.csv']+
                            [out/'models'/p/v/'selection.json' for p in PILOT for v in VARIANTS]))
            save_json(out/'frozen_models.json',frozen);frozen_hash=sha256(out/'frozen_models.json')
            del internal_inputs,internal_new,internal_base;gc.collect()
            status(stage='all_four_models_frozen_before_holdout')
            holdout_inputs={};replay=[]
            original_metrics=pd.read_csv(args.fusion_run/'metrics_by_panel_regime.csv')
            for panel in PILOT:
                if sha256(out/'frozen_models.json')!=frozen_hash: raise ValueError('Frozen models changed')
                data=source.data[panel];baseline=source.bundle(panel,'holdout',out/'frozen_models.json');new={}
                for variant in VARIANTS:
                    record=models[panel][variant]['model'];verify_model(Path(record['folder']))
                    model,_=model_for(config,'T1_balanced')
                    checkpoint,_=checkpoint_read(Path(record['folder']),'best.json',record['key'])
                    model.load_state_dict(checkpoint['state_dict']);model=model.to(device)
                    raw,pred_record=predictions(data,model,record,data.roles['holdout'],config,device,'holdout')
                    new[variant]=aligned(data.expected(data.roles['holdout']),raw)
                    models[panel][variant]['holdout_predictions']=pred_record
                    new[variant].to_parquet(out/'models'/panel/variant/'holdout_raw_predictions.parquet',index=False)
                    del model,checkpoint;gc.collect()
                    if device.type=='cuda': torch.cuda.empty_cache()
                holdout_inputs[panel]=method_inputs(baseline,new)
            detail,table,daily,preservation=write_scores(holdout_inputs,out,'holdout')
            for method,old_name in [('T0','T0'),('T1','T1'),('Current_equal','T01_equal'),('Current_fusion','Selected')]:
                for row in detail.loc[detail.method.eq(method)].itertuples(index=False):
                    saved=original_metrics.loc[original_metrics.method.eq(old_name)&original_metrics.panel.eq(row.panel)&original_metrics.regime.eq(row.regime)]
                    if len(saved)!=1: raise ValueError('Original control metric missing')
                    difference=max(abs(getattr(row,f)-float(saved.iloc[0][f])) for f in ['n','S_state','rmse_speed_kmh','rmse_flow_per_lane'])
                    if not args.smoke and difference>1e-9: raise ValueError('Original control scores changed')
                    replay.append(dict(panel=row.panel,method=method,regime=row.regime,max_difference=difference,full_scope=not args.smoke))
            pd.DataFrame(replay).to_csv(out/'control_score_replay.csv',index=False)
            intervals=[]
            for reference,candidate in [('T1','A_dynamic'),('T1','B_regime'),('Current_fusion','A_dynamic_fixed'),('Current_fusion','B_regime_fixed')]:
                pair=daily.loc[daily.method.isin([reference,candidate])].copy();pair.loc[pair.method.eq(reference),'method']='T0'
                for row in paired_bootstrap(pair,[candidate]): row['reference']=reference;intervals.append(row)
            pd.DataFrame(intervals).to_csv(out/'paired_calendar_bootstrap.csv',index=False)
            if sha256(out/'frozen_models.json')!=frozen_hash: raise ValueError('Frozen model manifest changed')
            save_json(out/'model_cache_usage.json',models)
            finish_source(source,out,samplers)
            verify(cfg['implementation'])
            save_json(out/'integrity_check.json',dict(passed=True,all_four_models_frozen_before_holdout=True,
                isolated_exact=all(x['isolated_exact'] for x in preservation.values()),fixed_formula_exact=True,
                control_scores_replayed=not args.smoke,full_dates=not args.smoke,
                scope='Two directions only; previously inspected local holdout; no ten-direction breakthrough claim'))
            save_json(out/'memory.json',dict(device=str(device),peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None))
            artifacts=snapshot([p for p in out.rglob('*') if p.is_file() and p.name not in ['execution.lock','status.json','COMPLETE.json','artifact_manifest.json','log_update.json']])
            save_json(out/'artifact_manifest.json',artifacts)
            save_json(out/'COMPLETE.json',dict(experiment=EXPERIMENT,smoke=args.smoke,integrity_passed=True,
                artifact_manifest_sha256=sha256(out/'artifact_manifest.json'),seconds=time.perf_counter()-started))
            result=log_results(out,args.project);status(stage='completed',log_status=result['status'],seconds=time.perf_counter()-started)
            print(table.to_string(index=False),flush=True);print('Log status:',result['status'],'\nTraining ablation complete:',out,flush=True)
        except Exception as error:
            status(stage='failed',error=repr(error));raise


def finish_source(source,out,samplers):
    source.finish()
    (out/'cache_usage.json').replace(out/'source_cache_usage.json')
    usage=read(out/'source_cache_usage.json');usage.pop('training_started',None)
    usage['scope']='Read-only source predictions/labels; new training is recorded separately in model_cache_usage.json'
    save_json(out/'source_cache_usage.json',usage)
    save_json(out/'observation_cache_usage.json',{p:s.observations.records for p,s in samplers.items()})
    # All newly published cache entries have relative manifests, and are validated
    # on use. Protect the observation manifests used in this run as well.
    observation_files=[Path(r['folder'])/'manifest.json' for s in samplers.values() for r in s.observations.records.values()]
    from task1.cache import FeatureCache
    for s in samplers.values():
        for r in s.observations.records.values():
            folder=Path(r['folder'])
            if sha256(folder/'manifest.json')!=r['manifest_sha256']: raise ValueError('Observation manifest changed during training')
            FeatureCache(folder.parent)._validate(folder,read(folder/'manifest.json')['identity'])
    save_json(out/'observation_cache_manifests.json',snapshot(observation_files))
