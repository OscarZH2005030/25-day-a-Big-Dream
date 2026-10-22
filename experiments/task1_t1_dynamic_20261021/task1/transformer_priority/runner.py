from __future__ import annotations

from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import gc
import json
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from . import data as dataset
from . import lightgbm,train
from .common import (PROJECT,HERE,COMMIT,METHODS,PILOT,FROZEN,KEY,TARGETS,SCALE,aligned,route,inside,
                     upstream,versions,implementation,frozen_plan,metric_rows,read,save_json,sha256,snapshot,device_for)
from ..metrics import aggregate
from ..cache import FeatureCache


def verify_snapshot(items):
    for item in items:
        path=inside(item['path'])
        if not path.is_file() or sha256(path)!=item['sha256']:
            raise ValueError('Protected source/input/artifact changed: '+str(path))


def summarize(detail):
    return aggregate(detail).merge(detail.groupby('method',as_index=False)[['rmse_speed_kmh','rmse_flow_per_lane']].mean(),on='method')


def context(data,dates):
    parts=[]
    for date in dates:
        raw,base=data.target_arrays(date);keys=data.keys(date).copy();ii=data.array(date,'ii')
        for channel,label in [('speed_kmh','gap'),('flow_vph','flow_gap')]:
            before=raw[:,data.names.index(channel+'_prev_gap_min')];after=raw[:,data.names.index(channel+'_next_gap_min')]
            span=before+after
            keys[label]=np.asarray(pd.cut(span,[0,15,30,60,np.inf],labels=['<=15min','15-30min','30-60min','>60min']).astype(str),dtype=object)
            keys.loc[~np.isfinite(span),label]='one_side_or_none'
        keys['date']=date;keys['isolated']=data.graph['isolated'][ii]
        keys['lanes']=data.exp.lanes[ii];keys['free_speed_kmh']=data.exp.static.free_speed_kmh.to_numpy()[ii]
        parts.append(keys)
    return pd.concat(parts,ignore_index=True)


def evaluate(data,prediction,ctx,truth,method,out):
    pred=aligned(ctx[KEY],prediction)
    values=np.column_stack([pred.speed_kmh.to_numpy(),pred.flow_vph.to_numpy()/ctx.lanes.to_numpy()])
    rows=metric_rows(truth,values,ctx.mask_regime.to_numpy(),method,data.panel,data.family)
    cells=ctx.copy();cells['speed_sq_error']=(values[:,0]-truth[:,0])**2;cells['flow_lane_sq_error']=(values[:,1]-truth[:,1])**2
    cells['congested']=truth[:,0]<=.6*ctx.free_speed_kmh.to_numpy()
    result={}
    for kind,columns in [('daily',['date','mask_regime']),('gap',['gap']),('flow_gap',['flow_gap']),
                         ('regime_gap',['mask_regime','gap']),('congestion',['congested']),('isolation',['isolated'])]:
        t=cells.groupby(columns,as_index=False).agg(n=('speed_sq_error','size'),speed_mse=('speed_sq_error','mean'),flow_lane_mse=('flow_lane_sq_error','mean'))
        t['method']=method;t['panel']=data.panel;t['family']=data.family
        t.to_csv(out/(method+'_'+kind+'_errors.csv'),index=False);result[kind]=t
    return rows,result


def bootstrap(daily,reference,candidate,replicates=10000):
    dates=sorted(daily.date.unique());panels=sorted(daily.panel.unique());rng=np.random.default_rng(20261018)
    counts=rng.multinomial(len(dates),np.full(len(dates),1/len(dates)),size=replicates)
    scores=[]
    for method in [reference,candidate]:
        values=np.zeros(replicates)
        for panel in panels:
            for regime in ['R1','R2','R3']:
                t=daily.loc[daily.method.eq(method)&daily.panel.eq(panel)&daily.mask_regime.eq(regime)].set_index('date').reindex(dates)
                n=t.n.fillna(0).to_numpy();total=counts@n
                if (total==0).any():
                    raise ValueError('Bootstrap replicate omitted an entire regime')
                speed=np.sqrt((counts@(t.speed_mse.fillna(0).to_numpy()*n))/total)
                flow=np.sqrt((counts@(t.flow_lane_mse.fillna(0).to_numpy()*n))/total)
                values+=(.54*np.maximum(0,1-speed/25)+.46*np.maximum(0,1-flow/600))/(len(panels)*3)
        scores.append(values)
    delta=scores[1]-scores[0];low,high=np.quantile(delta,[.025,.975])
    return dict(reference=reference,candidate=candidate,replicates=replicates,ci95_low=float(low),ci95_high=float(high),
                limitation='Paired calendar-date resampling of repeatedly inspected local holdout; descriptive, not independent-test evidence.')


def log_results(out):
    complete=read(out/'COMPLETE.json');cfg=read(out/'run_config.json')
    if complete['smoke']:
        result=dict(status='skipped_smoke');save_json(out/'log_update.json',result);return result
    if complete.get('experiment')!='transformer_priorities' or complete['methods']!=METHODS or complete['panels']!=PILOT:
        raise ValueError('Complete four-method two-panel pilot required for this logger')
    if sha256(out/'artifact_manifest.json')!=complete['artifact_manifest_sha256']:
        raise ValueError('Result manifest changed')
    manifest=read(out/'artifact_manifest.json')['artifacts']
    names={str(Path(item['path']).resolve().relative_to(out.resolve())) for item in manifest}
    required={'run_config.json','environment.json','comparison_methods.csv','metrics_by_panel_regime.csv',
              'integrity_check.json','preservation_checks.json','cache_usage.json','source_manifest.json','new_cache_manifest.json','truth_precision.csv'}
    required.update(str(Path(panel)/(method+'_predictions.parquet')) for panel in PILOT for method in METHODS)
    if not required.issubset(names):
        raise ValueError('Missing protected formal artifacts')
    verify_snapshot(manifest)
    if cfg['training']!=dict(FROZEN,eval_batch=2) or not complete['all_regimes_present'] or not complete['integrity_passed']:
        raise ValueError('Formal recipe/scope incomplete')
    checks=read(out/'integrity_check.json')
    if not all(checks[k] for k in ['passed','all_four_methods_same_targets','all_isolated_values_equal_shared_LightGBM','balanced_methods_paired_sampling']):
        raise ValueError('Formal invariant failed')
    table=pd.read_csv(out/'comparison_methods.csv')
    if set(table.method)!=set(METHODS) or table.method.duplicated().any() or not np.isfinite(table.S_state).all():
        raise ValueError('Missing formal method scores')
    detail=pd.read_csv(out/'metrics_by_panel_regime.csv')
    expected=pd.read_csv(HERE/'reference_routing_counts.csv').set_index('panel')
    preservation=read(out/'preservation_checks.json')
    for method in METHODS:
        for panel in PILOT:
            g=detail.loc[detail.method.eq(method)&detail.panel.eq(panel)]
            if len(g)!=3 or set(g.regime)!={'R1','R2','R3'} or g.n.sum()!=expected.loc[panel,'n']:
                raise ValueError('Formal template/regime coverage differs')
            item=preservation[panel+'/'+method]
            if not item['passed'] or not item['isolated_values_equal_LightGBM'] or not item['nonisolated_values_equal_network'] or item['rows']!=expected.loc[panel,'n'] or item['isolated_rows']!=expected.loc[panel,'fallback_n']:
                raise ValueError('Formal fallback preservation scope differs')
    rebuilt=summarize(detail).set_index('method')
    for row in table.itertuples(index=False):
        for name in ['S_state','rmse_speed_kmh','rmse_flow_per_lane']:
            if abs(getattr(row,name)-rebuilt.loc[row.method,name])>1e-10:
                raise ValueError('Formal macro aggregation differs')
        if abs(row.delta_vs_T0-(row.S_state-rebuilt.loc[METHODS[0],'S_state']))>1e-12:
            raise ValueError('Formal reference delta differs')
    log=PROJECT/'实验日志.txt';previous=log.read_text(encoding='utf-8-sig') if log.exists() else ''
    marker=f'实验编号：{out.name}；结果目录：{out}'
    if marker in previous:
        result=dict(status='already_logged',log_path=str(log));save_json(out/'log_update.json',result);return result
    lines=['',f'【图时空Transformer三优先路线两走廊正式试验｜{datetime.now(ZoneInfo("America/Los_Angeles")).date()}｜已完成】',marker,
           'D7_I405_N、D7_I405_S；历史28日、监督168日、内部22日、留出55日，与ad266f9协议一致。四组逐走廊独立训练，内部原始网络S选checkpoint。',
           'T0原采样；T1均衡覆盖；T2再加观测可靠性图消息权重；T3再加可学习邻居时间偏移。共同51特征/102维输入、插值残差、24步、64宽、3层、同预算，零物理惩罚。',
           '主表均为全局图孤立点两通道回退共享LightGBM；孤立点逐值一致、相同目标与公共参数初始化、T1/T2/T3配对采样检查通过。',
           '本机T0为公平主对照；历史0.936910是十方向均分，当前仅两方向一个家族，不用两路成绩宣称超过十路。留出已反复用于诊断，属于探索性本地验证。']
    for row in table.itertuples(index=False):
        lines.append(f'{row.method}：两路均分S={row.S_state:.9f}；对T0 ΔS={row.delta_vs_T0:+.9f}；速度宏RMSE={row.rmse_speed_kmh:.6f}；每车道流量宏RMSE={row.rmse_flow_per_lane:.6f}。')
    for row in pd.read_csv(out/'incremental_comparison.csv').itertuples(index=False):
        lines.append(f'{row.candidate} 对 {row.reference}：ΔS={row.delta_S_state:+.9f}。')
    lines+=['逐走廊/regime、逐日/缺口/流量缺口/拥堵、实际覆盖及采样摘要、注意力/lag分布、显存/耗时与hash见结果目录。','']
    with log.open('a',encoding='utf-8',newline='') as stream:
        stream.write('\n'.join(lines)+'\n')
    result=dict(status='appended',log_path=str(log));save_json(out/'log_update.json',result);return result


def run(args):
    started=time.perf_counter();root=inside(args.release_root);out=inside(args.output);cache_root=inside(args.cache_dir)
    plan=frozen_plan();panels=args.panels
    if len(set(panels))!=len(panels) or any(p not in plan['panels'] for p in panels):
        raise ValueError('Invalid panel list')
    if panels!=PILOT:
        raise ValueError('This authorized pilot fixes D7_I405_N and D7_I405_S; expand only after reviewing pilot results')
    config=dict(FROZEN,eval_batch=2)
    if args.smoke:
        config.update(epochs=2,steps_per_epoch=4,patience=2)
    device=device_for(args.device);torch.set_num_threads(config['threads'])
    source_closure=implementation()+snapshot([HERE/n for n in ['upstream_sha256.json','reference_split.json','reference_feature_names.json','reference_routing_counts.csv']])
    cfg=dict(experiment='transformer_priorities',commit=COMMIT,panels=panels,methods=METHODS,smoke=args.smoke,
             release_root=str(root),cache_dir=str(cache_root),training=config,split=plan,device=str(device),
             implementation=source_closure,versions=versions(),reference_score=0.9369103120317692,
             reference_scope='Historical10-direction local mean; this run has2 directions only')
    out.mkdir(parents=True,exist_ok=True);cache_root.mkdir(parents=True,exist_ok=True)
    with upstream().lock(cache_root/'execution.lock'),upstream().lock(out/'queue.lock'):
        if (out/'run_config.json').exists():
            if not args.resume or read(out/'run_config.json')!=cfg:
                raise ValueError('Output configuration/source changed; use unchanged --resume or a new output')
        else:
            save_json(out/'run_config.json',cfg)
        if (out/'COMPLETE.json').exists():
            complete=read(out/'COMPLETE.json')
            if sha256(out/'artifact_manifest.json')!=complete['artifact_manifest_sha256']:
                raise ValueError('Completed artifact manifest changed')
            verify_snapshot(read(out/'artifact_manifest.json')['artifacts']);verify_snapshot(read(out/'source_manifest.json'))
            print(log_results(out),flush=True);return
        state=dict(state='running',completed_models=[],panel=None,method=None,stage='starting')
        def status(**values):
            state.update(values);state['updated_utc']=datetime.now(timezone.utc).isoformat();save_json(out/'status.json',state)
        env=dict(python=sys.version,versions=versions(),source_files=cfg['implementation'],
                 device=str(device),cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                 cuda_total_gib=torch.cuda.get_device_properties(device).total_memory/2**30 if device.type=='cuda' else None)
        save_json(out/'environment.json',env)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        detail=[];raw_detail=[];daily=[];sources=[];cache_records={};all_preserved={};paired={};initial=[];cache_files=[];precision=[]
        try:
            for panel in panels:
                status(panel=panel,method=None,stage='preparing_features')
                destination=out/panel;destination.mkdir(exist_ok=True)
                ds=dataset.prepare(root,panel,cache_root,plan,args.smoke)
                save_json(destination/'feature_cache.json',ds.provenance)
                cache_files += [cache_root/ds.provenance[k]/'manifest.json' for k in ['template_key','history_key','normalization_key']]
                cache_files += [Path(record['folder'])/'manifest.json' for record in ds.records.values()]
                ref=pd.read_csv(HERE/'reference_routing_counts.csv').set_index('panel').loc[panel]
                if ds.n!=ref.nodes or int(ds.graph['isolated'].sum())!=ref.isolated_nodes:
                    raise ValueError('Original node graph/isolation differs')
                dates=ds.roles['holdout'];expected=ds.expected(dates);ctx=context(ds,dates)
                if not args.smoke and (len(expected)!=ref.n or int(ctx.isolated.sum())!=ref.fallback_n):
                    raise ValueError('Full original holdout target/isolation scope differs')
                if args.action=='prepare':
                    save_json(destination/'preflight.json',dict(passed=True,nodes=ds.n,roles=ds.roles,
                        training_rows=sum(ds.records[d]['rows'] for d in ds.roles['training']),holdout_rows=len(expected),
                        isolated_rows=int(ctx.isolated.sum()),holdout_labels_read=False,formal_training_started=False))
                    sources+=ds.provenance['inputs'];ds.array.cache_clear();ds.keys.cache_clear();del ds;gc.collect();continue
                # Preload only training/internal labels, then freeze their source identities before fitting.
                for purpose in ['training','internal']:
                    for date in ds.roles[purpose]:
                        ds.labels(date,purpose)
                signature=dict(features={d:r['key'] for d,r in ds.records.items()},normalization=ds.provenance['normalization_key'],
                               labels={d:r['key'] for d,r in ds.label_records.items()},roles=ds.roles)
                frozen=destination/'data_signature.json'
                if frozen.exists() and read(frozen)!=signature:
                    raise ValueError('Data changed since this run started')
                save_json(frozen,signature)
                status(panel=panel,stage='shared_lightgbm')
                trees,tree_record=lightgbm.fit(ds,args.smoke,config['threads'])
                lgb_pred,lgb_prediction_record=lightgbm.predictions(ds,trees,tree_record,dates,config['threads'])
                lgb_pred=aligned(expected,lgb_pred);lgb_pred.to_parquet(destination/'LightGBM_predictions.parquet',index=False)
                cache_records[panel]=dict(lightgbm=tree_record,lightgbm_predictions=lgb_prediction_record,models={})
                truth=None
                source_identity=dict(implementation=cfg['implementation'],versions=cfg['versions'],
                    device=str(device),cuda_device=env['cuda_device'],training_internal_labels=signature['labels'])
                for method in METHODS:
                    status(panel=panel,method=method,stage='model_starting')
                    ds.allow_holdout=False
                    model,model_record=train.fit(ds,config,method,device,cache_root,source_identity,status)
                    initial.append(model_record['selection']['common_initial_sha256'])
                    history=pd.read_csv(Path(model_record['folder'])/'training_history.csv')
                    history.to_csv(destination/(method+'_training_history.csv'),index=False)
                    if method in METHODS[1:]:
                        paired[(panel,method)]=history[['epoch','sampling_sha256']]
                    save_json(destination/(method+'_selection.json'),model_record['selection'])
                    status(panel=panel,method=method,stage='final_prediction')
                    raw,pred_record=train.cached_predictions(ds,model,model_record,dates,config,device,'holdout')
                    raw=aligned(expected,raw)
                    combined=route(expected,raw,lgb_pred,ctx.isolated.to_numpy())
                    raw.to_parquet(destination/(method+'_raw_predictions.parquet'),index=False)
                    combined.to_parquet(destination/(method+'_predictions.parquet'),index=False)
                    all_preserved[panel+'/'+method]=dict(passed=True,rows=len(expected),isolated_rows=int(ctx.isolated.sum()),
                        isolated_values_equal_LightGBM=True,nonisolated_values_equal_network=True)
                    ds.allow_holdout=True
                    if truth is None:
                        truth=ds.truth(dates,'holdout')
                    rows,groups=evaluate(ds,combined,ctx,truth,method,destination);detail+=rows;daily.append(groups['daily'])
                    rounded=(truth/SCALE).astype(np.float32).astype(np.float64)*SCALE
                    values=np.column_stack([combined.speed_kmh.to_numpy(),combined.flow_vph.to_numpy()/ctx.lanes.to_numpy()])
                    rounded_rows=metric_rows(rounded,values,ctx.mask_regime.to_numpy(),method,ds.panel,ds.family)
                    precision.append(dict(panel=panel,method=method,raw_truth_S_state=float(np.mean([r['S_state'] for r in rows])),
                        original_float32_restored_truth_S_state=float(np.mean([r['S_state'] for r in rounded_rows])),
                        max_truth_speed_difference=float(np.abs(truth[:,0]-rounded[:,0]).max()),
                        max_truth_flow_lane_difference=float(np.abs(truth[:,1]-rounded[:,1]).max())))
                    rows,_=evaluate(ds,raw,ctx,truth,method+'_raw',destination);raw_detail+=rows
                    cache_records[panel]['models'][method]=dict(model=model_record,predictions=pred_record)
                    state['completed_models'].append(panel+'/'+method)
                    status(panel=panel,method=method,stage='model_complete')
                    pd.DataFrame(detail).to_csv(out/'partial_metrics_by_panel_regime.csv',index=False)
                    summarize(pd.DataFrame(detail)).to_csv(out/'partial_metrics_summary.csv',index=False)
                    print('[completed]',panel,method,'routed_S=',np.mean([r['S_state'] for r in detail if r['panel']==panel and r['method']==method]),flush=True)
                    del model;gc.collect()
                    if device.type=='cuda':
                        torch.cuda.empty_cache()
                sources+=ds.provenance['inputs']
                for item in ds.label_records.values():
                    sources+=item['source']
                    cache_files.append(Path(item['folder'])/'manifest.json')
                for key in ['lightgbm','lightgbm_predictions']:
                    cache_files.append(Path(cache_records[panel][key]['folder'])/'manifest.json')
                for item in cache_records[panel]['models'].values():
                    model_folder=Path(item['model']['folder'])
                    cache_files += [model_folder/n for n in ['identity.json','COMPLETE.json','best.json',item['model']['best']['name']]]
                    cache_files.append(Path(item['predictions']['folder'])/'manifest.json')
                save_json(destination/'label_access.json',dict(records=ds.label_records,fit_dates=ds.roles['training'],internal_dates=ds.roles['internal'],holdout_scored_dates=dates))
                ds.array.cache_clear();ds.keys.cache_clear();del ds,trees,truth;gc.collect()
            verify_snapshot(cfg['implementation']);verify_snapshot(sources)
            save_json(out/'source_manifest.json',sources)
            if args.action=='prepare':
                save_json(out/'PREFLIGHT.json',dict(passed=True,smoke=args.smoke,panels=panels,formal_training_started=False,holdout_labels_read=False,seconds=time.perf_counter()-started))
                status(state='prepared',panel=None,method=None,stage='complete');print('Full preflight complete.',flush=True);return
            details=pd.DataFrame(detail);table=summarize(details)
            base=float(table.loc[table.method.eq(METHODS[0]),'S_state'].iloc[0]);table['delta_vs_T0']=table.S_state-base
            details.to_csv(out/'metrics_by_panel_regime.csv',index=False)
            table.to_csv(out/'comparison_methods.csv',index=False)
            pd.DataFrame(raw_detail).to_csv(out/'raw_metrics_by_panel_regime.csv',index=False)
            summarize(pd.DataFrame(raw_detail)).to_csv(out/'raw_comparison_methods.csv',index=False)
            daily_table=pd.concat(daily,ignore_index=True);daily_table.to_csv(out/'daily_errors.csv',index=False)
            comparisons=[];cis=[];values=table.set_index('method').S_state
            for reference,candidate in [(METHODS[0],m) for m in METHODS[1:]]+[(METHODS[1],METHODS[2]),(METHODS[2],METHODS[3])]:
                comparisons.append(dict(reference=reference,candidate=candidate,delta_S_state=float(values[candidate]-values[reference])))
                if not args.smoke:
                    cis.append(bootstrap(daily_table,reference,candidate))
            pd.DataFrame(comparisons).to_csv(out/'incremental_comparison.csv',index=False)
            pd.DataFrame(cis).to_csv(out/'paired_calendar_bootstrap.csv',index=False)
            for panel in panels:
                for method in METHODS[2:]:
                    a,b=paired[(panel,METHODS[1])],paired[(panel,method)]
                    n=min(len(a),len(b))
                    if not a.sampling_sha256.iloc[:n].tolist()==b.sampling_sha256.iloc[:n].tolist():
                        raise ValueError('Balanced model sampling sequences differ')
            if len(set(initial))!=1:
                raise ValueError('Shared initial parameters differ across methods/panels')
            save_json(out/'cache_usage.json',cache_records)
            for path in sorted(set(cache_files)):
                if path.name=='manifest.json':
                    manifest=read(path)
                    FeatureCache(path.parent.parent)._validate(path.parent,manifest['identity'])
            save_json(out/'new_cache_manifest.json',snapshot(cache_files))
            pd.DataFrame(precision).to_csv(out/'truth_precision.csv',index=False)
            save_json(out/'preservation_checks.json',all_preserved)
            save_json(out/'integrity_check.json',dict(passed=True,all_four_methods_same_targets=True,all_isolated_values_equal_shared_LightGBM=True,
                balanced_methods_paired_sampling=True,common_initialization_equal=True,no_holdout_model_selection=True,reference_protocol=True))
            save_json(out/'memory.json',dict(device=env['cuda_device'],peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else None,note='Process allocator peak, not whole GPU usage'))
            if (out/'FAILED.json').exists():
                save_json(out/'RECOVERED.json',read(out/'FAILED.json'))
                (out/'FAILED.json').unlink()
            status(state='complete',panel=None,method=None,stage='complete')
            artifacts=[p for p in out.rglob('*') if p.is_file() and p.name not in ['queue.lock','artifact_manifest.json','COMPLETE.json','log_update.json']
                       and p.suffix not in ['.log']]
            save_json(out/'artifact_manifest.json',dict(artifacts=snapshot(artifacts)))
            save_json(out/'COMPLETE.json',dict(experiment='transformer_priorities',smoke=args.smoke,panels=panels,methods=METHODS,
                all_regimes_present=bool(table.all_regimes_present.all()),integrity_passed=True,artifact_manifest_sha256=sha256(out/'artifact_manifest.json'),
                scope='Two-direction pilot; no full10-direction claim',seconds=time.perf_counter()-started))
            print(table.to_string(index=False),flush=True);print(log_results(out),flush=True)
        except BaseException as error:
            status(state='failed',error=str(error));save_json(out/'FAILED.json',dict(error=repr(error),state=state));raise
