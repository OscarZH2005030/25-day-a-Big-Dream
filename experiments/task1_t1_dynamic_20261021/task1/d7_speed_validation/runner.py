from __future__ import annotations

from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from ..cache import snapshot
from ..data import KEY,TARGETS,day_files,network,save_json,sha256
from ..fusion_experiments import aligned_prediction
from ..grin.hybrid import check_scores
from ..metrics import score,aggregate
from ..pipeline import truth_rows,panel_families
from ..traffic_wave.fit import ordered_key_hash
from ..wave_neural.cache import NeuralCache
from ..wave_neural.runner import seed
from ..wave_context.runner import memory
from ..speed_fusion.rules import CONTROL,NEURAL,WEIGHTS,PRIMARY,METHODS,RULE,context_for,fuse_speed,audit
from ..speed_fusion.runner import comparison,export_errors
from .reference import PROJECT,PANEL,TRAIN_ROWS,EVAL_ROWS,OLD_METHOD,SPLIT,frozen_recipe,load_sources,control_reference
from .stages import resources,wave_predictions,daily_inputs,neural_checks,neural_predictions,prepare_probe,implementation
from .log import append_log


def scope(source,smoke):
    train_dates=source['grid_train']['metadata']['dates'][:14] if smoke else source['grid_train']['metadata']['dates']
    eval_dates=source['grid_eval']['metadata']['dates'][:3] if smoke else source['grid_eval']['metadata']['dates']
    keys=source['keys'].loc[source['keys'].timestamp.str[:10].isin(train_dates)].reset_index(drop=True)
    expected=source['expected'].loc[source['expected'].timestamp.str[:10].isin(eval_dates)].reset_index(drop=True)
    context=context_for(expected,source['diagnostics'].loc[source['diagnostics'].timestamp.str[:10].isin(eval_dates)])
    return train_dates,eval_dates,keys,expected,context


def old_controls(source,expected,truth,net,family,dates,smoke):
    rows=[];checks=[];predictions={}
    for method,key in [('Fusion_gap60mean','baseline'),('GRIN','grin'),(OLD_METHOD,'current')]:
        prediction=aligned_prediction(expected,source[key].loc[source[key].timestamp.str[:10].isin(dates)])
        part,_=score(prediction,truth,net,family,method);rows+=part
        check=check_scores(control_reference(source,method,dates,smoke),pd.DataFrame(part),method)
        if not check['passed']:raise ValueError('D7 frozen original control failed: '+method)
        checks.append(check);predictions[method]=prediction
    return rows,checks,predictions


def with_i5(table,reference):
    a=reference[['method','S_state','delta_S_state_vs_current_best']].rename(columns={'S_state':'I5_S_state','delta_S_state_vs_current_best':'I5_delta_S'})
    b=table[['method','S_state','delta_S_state_vs_current_best']].rename(columns={'S_state':'D7_S_state','delta_S_state_vs_current_best':'D7_delta_S'})
    result=a.merge(b,on='method',validate='one_to_one')
    if set(result.method)!=set(METHODS):raise ValueError('I5/D7 method pairing differs')
    result['two_north_corridor_family_mean_S_state']=.5*(result.I5_S_state+result.D7_S_state)
    result['scope']='D12_I5_N and D7_I405_N only'
    return result


def run(args):
    started=time.perf_counter();output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new empty output directory')
    recipe=frozen_recipe(args)
    cfg=dict(SPLIT,**recipe['wave_parameters'],**recipe['neural'])
    cfg.update(panel=PANEL,sample_per_day=1000,rounds=50 if args.smoke else 350,epochs=2 if args.smoke else 30,
        steps_per_epoch=4 if args.smoke else 100,smoke=args.smoke,no_log=args.no_log,device=args.device,
        output=str(output),cache_dir=str(args.cache_dir),baseline_run=str(args.baseline_run),hybrid_run=str(args.hybrid_run),
        action=args.action,weights=WEIGHTS,primary_method=PRIMARY,rule=RULE,
        reference_definition='Frozen I5 wave-enhanced recipe re-fitted on D7; original D7 baseline/GRIN separately reported')
    seed(cfg);torch.set_num_threads(cfg['threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; explicitly select cpu or check existing .venv-grin CUDA environment')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    print(f'[D7 validation] device={device}; mode={"smoke" if args.smoke else "formal"}; frozen I5 recipe; old baseline/GRIN read-only',flush=True)
    source=load_sources(args,recipe);source['recipe']=recipe
    train_dates,eval_dates,train_keys,expected,context=scope(source,args.smoke)
    nodes=pd.DataFrame(source['grid_train']['metadata']['nodes']);net=network(source['root'],PANEL);family=panel_families(source['root'])[PANEL]
    truth=truth_rows(source['root'],PANEL,expected)
    controls,checks,old=old_controls(source,expected,truth,net,family,eval_dates,args.smoke)
    print(f'[D7 sources] exact samples: {len(train_keys)}/{TRAIN_ROWS}; eval={len(expected)}/{EVAL_ROWS}; nodes={len(nodes)}; three old controls reproduced',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    cfg.update(actual_fit_dates=train_dates,actual_eval_dates=eval_dates,release_root=str(source['root']))
    save_json(output/'run_config.json',cfg)
    save_json(output/'frozen_recipe.json',{k:v for k,v in recipe.items() if k not in ['provenance','i5_comparison']})
    save_json(output/'reference_check.json',dict(passed=True,checks=checks,old_baseline_GRIN_exact_from_saved_constituents=True,
        original_baseline_and_GRIN_retrained=False,original_GRIN_inference_run=False,subset=args.smoke))
    save_json(output/'sample_check.json',dict(passed=True,formal_training_rows=TRAIN_ROWS,formal_evaluation_rows=EVAL_ROWS,
        training_rows=len(train_keys),evaluation_rows=len(expected),ordered_training_keys_sha256=ordered_key_hash(train_keys),
        exact_original_baseline_GRIN_targets=True,train_days=len(train_dates),eval_days=len(eval_dates),nodes=len(nodes)))
    events=dict(wave_models_trained=False,wave_model_cache_accessed=False,wave_prediction_inference=False,wave_days_built=0,
        neural_model_trained=False,neural_model_cache_accessed=False,neural_prediction_inference=False,input_days_built=0,
        original_baseline_retrained=False,original_GRIN_retrained=False,original_GRIN_inference=False)
    res=resources(source,cfg,nodes,net)
    usage=dict(original_feature_and_grid_caches_reused=True,original_baseline_predictions_reused=True,original_GRIN_predictions_reused=True,
        kernel=dict(key=res['kernel']['key'],hit=res['kernel_hit']),statistics=dict(key=res['stats']['key'],hit=res['stats_hit']),
        static_attributes=dict(key=res['attrs']['key'],hit=res['attrs_hit']),events=events)
    save_json(output/'cache_usage.json',usage)
    save_json(output/'environment.json',dict(python=platform.python_version(),device=str(device),source_files=implementation(),
        cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None))
    truth_files=day_files(source['root'],PANEL,'train',False);truth_hashes=snapshot(truth_files[d] for d in eval_dates)
    provenance=source['provenance']+[res[name]/'manifest.json' for name in ['kernel_folder','stats_folder','attrs_folder']]+[truth_files[d] for d in eval_dates]
    save_json(output/'source_manifest.json',snapshot(provenance))
    if args.action=='prepare':
        probes=prepare_probe(cfg,source,res,nodes,device,events)
        save_json(output/'neural_checks.json',probes);save_json(output/'cache_usage.json',usage)
        table=aggregate(pd.DataFrame(controls));scores={r.method:float(r.S_state) for r in table.itertuples(index=False)}
        save_json(output/'PREFLIGHT.json',dict(passed=True,training_rows=len(train_keys),evaluation_rows=len(expected),nodes=len(nodes),
            train_days=len(train_dates),eval_days=len(eval_dates),gated_rows=int(context.blend_speed.sum()),old_control_scores=scores,
            formal_models_trained=False,formal_predictions_generated=False,random_initialization_probe=True))
        save_json(output/'memory.json',memory(device));save_json(output/'log_update.json',dict(status='skipped_prepare'))
        print(f'Preflight passed: {len(train_keys)} train, {len(expected)} eval, {len(nodes)} nodes; controls={scores}; no formal models trained. {output}',flush=True)
        return
    cache=NeuralCache(args.cache_dir)
    current,wave_record=wave_predictions(cache,cfg,source,res,nodes,train_keys,expected,context,old['GRIN'],train_dates,eval_dates,device,events)
    usage['wave_prediction']=wave_record;save_json(output/'cache_usage.json',usage)
    inputs=daily_inputs(source,res,nodes,wave_record,events);usage['training_days']=inputs['training'];usage['evaluation_days']=inputs['evaluation']
    adjacency=np.load(source['grid_train_folder']/'adjacency.npy',allow_pickle=False)
    probes=neural_checks(source,res,inputs['training'],adjacency,cfg,device);save_json(output/'neural_checks.json',probes)
    neural,neural_record=neural_predictions(cache,cfg,source,res,inputs,adjacency,nodes,train_keys,expected,truth,net,family,truth_hashes,device,events)
    usage['neural_prediction']=neural_record;save_json(output/'cache_usage.json',usage)
    for method,prediction in [(CONTROL,current),(NEURAL,neural)]:
        part,_=score(prediction,truth,net,family,method);controls+=part
    sources=pd.DataFrame(controls)
    source_table=aggregate(sources).merge(sources.groupby('method',as_index=False)[['rmse_speed_kmh','rmse_flow_per_lane']].mean(),on='method')
    source_table.to_csv(output/'source_controls.csv',index=False);sources.to_csv(output/'source_controls_by_regime.csv',index=False)
    old[OLD_METHOD].to_parquet(output/'Frozen_D7_baseline_GRIN_predictions.parquet',index=False)
    neural.to_parquet(output/(NEURAL+'_predictions.parquet'),index=False)
    pd.read_csv(Path(neural_record['folder'])/'training_history.csv').to_csv(output/'neural_training_history.csv',index=False)
    save_json(output/'neural_selection.json',neural_record['metadata']['model'])
    context.to_parquet(output/'fusion_gate.parquet',index=False)
    context.groupby(['mask_regime','gap','blend_speed'],as_index=False).size().rename(columns={'size':'n'}).to_csv(output/'gate_counts.csv',index=False)
    rows=[];regime_gap=[];preserved={}
    for method,alpha in [(CONTROL,0.),*WEIGHTS.items()]:
        prediction=current.copy() if method==CONTROL else fuse_speed(expected,current,neural,context,alpha)
        preserved[method]=audit(expected,current,neural,prediction,context,alpha)
        prediction.to_parquet(output/(method+'_predictions.parquet'),index=False)
        local,extra=export_errors(output,method,prediction,truth,net,family,context);rows+=local;regime_gap.append(extra)
    detail=pd.DataFrame(rows);table=comparison(detail)
    detail.to_csv(output/'metrics_by_panel_regime.csv',index=False);aggregate(detail).to_csv(output/'metrics_summary.csv',index=False)
    table.to_csv(output/'comparison_methods.csv',index=False);pd.concat(regime_gap).to_csv(output/'regime_gap_errors.csv',index=False)
    with_i5(table,recipe['i5_comparison']).to_csv(output/'comparison_with_i5.csv',index=False)
    save_json(output/'preservation_checks.json',dict(passed=True,reference=CONTROL,methods=preserved,
        new_reference_long_gap_equals_saved_GRIN=wave_record['metadata']['grin_preservation']))
    save_json(output/'integrity_check.json',dict(passed=True,all_four_methods_same_targets=True,all_flow_values_exact=True,
        all_non_gated_speed_exact=True,original_baseline_and_GRIN_reused=True,no_source_prediction_features_in_training=True,
        frozen_I5_recipe=True,primary_method=PRIMARY,reference=CONTROL,original_D7_combo_reported_separately=True))
    save_json(output/'memory.json',memory(device))
    parent_paths=[Path(r['folder'])/'manifest.json' for r in [wave_record,neural_record]]
    parent_paths += [Path(r['folder'])/'manifest.json' for r in inputs['training']+inputs['evaluation']]
    save_json(output/'new_cache_manifest.json',snapshot(parent_paths))
    save_json(output/'artifact_manifest.json',dict(artifacts=snapshot(f for f in output.iterdir() if f.is_file())))
    save_json(output/'COMPLETE.json',dict(experiment='d7_speed_validation',panel=PANEL,smoke=args.smoke,methods=METHODS,
        training_rows=len(train_keys),evaluation_rows=len(expected),gated_rows=int(context.blend_speed.sum()),reference_passed=True,
        pairing_passed=True,integrity_passed=True,all_regimes_present=bool(table.all_regimes_present.all()),
        artifact_manifest_sha256=sha256(output/'artifact_manifest.json'),seconds=time.perf_counter()-started))
    try:status=append_log(output,PROJECT/'实验日志.txt')
    except (ValueError,OSError) as error:
        save_json(output/'log_update.json',dict(status='failed',error=str(error)))
        raise RuntimeError('D7 comparison complete; retry log without training') from error
    print(table.to_string(index=False),flush=True);print(f'Log status: {status["status"]}. D7 validation complete: {output}',flush=True)
