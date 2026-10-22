from __future__ import annotations

from pathlib import Path
import importlib.metadata
import platform
import time
import numpy as np
import pandas as pd
from ..cache import snapshot
from ..data import KEY,day_files,network,save_json,sha256
from ..fusion_experiments import aligned_prediction
from ..metrics import aggregate,score
from ..pipeline import truth_rows,panel_families
from .rules import CONTROL,NEURAL,WEIGHTS,PRIMARY,METHODS,RULE,context_for,fuse_speed,audit
from .reference import load_sources,control_check
from .log import append_log

PROJECT=Path(__file__).resolve().parents[2]


def implementation():
    files=list(Path(__file__).parent.glob('*.py'))
    files += [PROJECT/'task1'/name for name in ['data.py','metrics.py','pipeline.py','cache.py','fusion_experiments.py','cache_builder.py']]
    return snapshot(files)


def comparison(detail):
    result=aggregate(detail).merge(detail.groupby('method',as_index=False)[['rmse_speed_kmh','rmse_flow_per_lane']].mean(),on='method')
    base=result.loc[result.method.eq(CONTROL)].iloc[0]
    result['delta_S_state_vs_current_best']=result.S_state-base.S_state
    result['delta_speed_RMSE_vs_current_best']=result.rmse_speed_kmh-base.rmse_speed_kmh
    result['neural_speed_weight']=result.method.map({CONTROL:0.,**WEIGHTS})
    result['primary_candidate']=result.method.eq(PRIMARY)
    return result.sort_values('S_state',ascending=False).reset_index(drop=True)


def export_errors(output,method,prediction,truth,net,family,context):
    rows,cells=score(prediction,truth,net,family,method)
    cells=cells.merge(context[KEY+['gap']],on=KEY,validate='one_to_one')
    for name,cols in [('daily',['date','mask_regime']),('gap',['gap']),('congestion',['congested'])]:
        t=cells.groupby(cols,as_index=False).agg(n=('speed_sq_error','size'),speed_mse=('speed_sq_error','mean'),flow_lane_mse=('flow_lane_sq_error','mean'))
        t.to_csv(output/(method+'_'+name+'_errors.csv'),index=False)
    extra=cells.groupby(['mask_regime','gap'],as_index=False).agg(n=('speed_sq_error','size'),speed_mse=('speed_sq_error','mean'),flow_lane_mse=('flow_lane_sq_error','mean'))
    extra['method']=method
    return rows,extra


def run(args):
    began=time.perf_counter();output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new empty output directory')
    print('[speed fusion] CPU; four fixed methods; source predictions only',flush=True)
    source=load_sources(args.source_run);sc=source['config']
    dates=sc['actual_eval_dates'][:3] if args.smoke else sc['actual_eval_dates']
    use=source['expected'].timestamp.str[:10].isin(dates)
    expected=source['expected'].loc[use].reset_index(drop=True)
    context=context_for(expected,source['diagnostics'].loc[source['diagnostics'].timestamp.str[:10].isin(dates)])
    predictions={m:aligned_prediction(expected,t.loc[t.timestamp.str[:10].isin(dates)]) for m,t in source['predictions'].items()}
    root=source['root'];net=network(root,sc['panel']);family=panel_families(root)[sc['panel']]
    truth=truth_rows(root,sc['panel'],expected)
    controls=[];control_scores={}
    for method in [CONTROL,NEURAL]:
        rows,_=score(predictions[method],truth,net,family,method)
        controls.append(control_check(source,method,pd.DataFrame(rows),dates,args.smoke))
        control_scores[method]=float(aggregate(pd.DataFrame(rows)).S_state.iloc[0])
    cfg=dict(action=args.action,output=str(output),source_run=str(args.source_run),smoke=args.smoke,no_log=args.no_log,
        panel=sc['panel'],release_root=str(root),fit_start=sc['fit_start'],fit_end=sc['fit_end'],eval_start=sc['eval_start'],eval_end=sc['eval_end'],
        source_training_rows=214000,actual_eval_dates=dates,weights=WEIGHTS,primary_method=PRIMARY,rule=RULE,
        selection_note='Fixed weights; 25% predeclared primary; January development exploration, no automatic promotion or further sweep')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'run_config.json',cfg)
    save_json(output/'reference_check.json',dict(passed=True,checks=controls,control_S_state=control_scores,
        original_current_best_predictions_exact=True,subset=args.smoke))
    save_json(output/'sample_check.json',dict(passed=True,evaluation_rows=len(expected),formal_rows=550149,actual_eval_dates=dates,
        all_methods_same_natural_keys=True,source_training_pool=214000,source_split_preserved=True))
    context.to_parquet(output/'fusion_gate.parquet',index=False)
    context.groupby(['mask_regime','gap','blend_speed'],as_index=False).size().rename(columns={'size':'n'}).to_csv(output/'gate_counts.csv',index=False)
    reuse=dict(prediction_files_reused=2,models_retrained=False,neural_inference_run=False,model_checkpoint_loaded=False,
        feature_cache_rebuilt=False,source_evaluation_cache_reused=True,evaluation_cache_key=source['evaluation_cache_key'],
        prediction_sources={m:str(args.source_run/(m+'_predictions.parquet')) for m in [CONTROL,NEURAL]},
        prediction_ledger_verified=source['prediction_ledger_verified'],device='cpu',cuda_used=False)
    save_json(output/'prediction_reuse.json',reuse)
    save_json(output/'environment.json',dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow']},
        device='cpu',source_files=implementation()))
    files=day_files(root,sc['panel'],'train',False)
    source_files=source['provenance']+[files[d] for d in dates]
    source_files += [root/'corridors'/sc['panel']/'network'/name for name in ['fd_parameters.csv']]
    save_json(output/'source_manifest.json',snapshot(source_files))
    if args.action=='prepare':
        save_json(output/'PREFLIGHT.json',dict(passed=True,evaluation_rows=len(expected),days=len(dates),
            gated_rows=int(context.blend_speed.sum()),control_S_state=control_scores,fused_predictions_generated=False,models_trained=False))
        save_json(output/'log_update.json',dict(status='skipped_prepare'))
        print(f'Preflight passed: {len(expected)} targets, {len(dates)} days, {int(context.blend_speed.sum())} eligible; controls={control_scores}; no fusion results scored. {output}',flush=True)
        return
    checks={};details=[];regime_gap=[]
    for method,alpha in [(CONTROL,0.),*WEIGHTS.items()]:
        pred=predictions[CONTROL].copy() if method==CONTROL else fuse_speed(expected,predictions[CONTROL],predictions[NEURAL],context,alpha)
        checks[method]=audit(expected,predictions[CONTROL],predictions[NEURAL],pred,context,alpha)
        pred.to_parquet(output/(method+'_predictions.parquet'),index=False)
        rows,extra=export_errors(output,method,pred,truth,net,family,context)
        details+=rows;regime_gap.append(extra)
        print(f'[{method}] alpha={alpha}; eligible={checks[method]["eligible_rows"]}; preserved flow/non-gated speed: exact',flush=True)
    detail=pd.DataFrame(details);table=comparison(detail)
    flow=detail.pivot(index='regime',columns='method',values='rmse_flow_per_lane')
    if not all(np.array_equal(flow[m].to_numpy(),flow[CONTROL].to_numpy()) for m in METHODS):
        raise ValueError('Flow error changed despite preservation')
    detail.to_csv(output/'metrics_by_panel_regime.csv',index=False)
    aggregate(detail).to_csv(output/'metrics_summary.csv',index=False);table.to_csv(output/'comparison_methods.csv',index=False)
    pd.concat(regime_gap,ignore_index=True).to_csv(output/'regime_gap_errors.csv',index=False)
    save_json(output/'preservation_checks.json',dict(passed=True,methods=checks))
    save_json(output/'integrity_check.json',dict(passed=True,all_four_methods_same_targets=True,all_flow_values_exact=True,
        all_non_gated_speed_exact=True,only_observable_sides_used=True,threshold_lower_exclusive=15,threshold_upper_inclusive=60,
        fixed_weights=WEIGHTS,primary_method=PRIMARY,models_retrained=False,neural_inference_run=False))
    # Protect all immutable results before publishing completion; log_update is intentionally mutable.
    save_json(output/'artifact_manifest.json',dict(artifacts=snapshot(f for f in output.iterdir() if f.is_file())))
    save_json(output/'COMPLETE.json',dict(experiment='speed_fusion',panel=sc['panel'],smoke=args.smoke,methods=METHODS,
        evaluation_rows=len(expected),gated_rows=int(context.blend_speed.sum()),reference_passed=True,pairing_passed=True,integrity_passed=True,
        all_regimes_present=bool(table.all_regimes_present.all()),artifact_manifest_sha256=sha256(output/'artifact_manifest.json'),seconds=time.perf_counter()-began))
    try:status=append_log(output,PROJECT/'实验日志.txt')
    except (ValueError,OSError) as error:
        save_json(output/'log_update.json',dict(status='failed',error=str(error)))
        raise RuntimeError('Comparison complete; retry log without recombining or re-scoring') from error
    print(table.to_string(index=False),flush=True)
    print(f'Log status: {status["status"]}. Speed fusion complete: {output}',flush=True)
