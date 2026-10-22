from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..cache import snapshot
from ..data import KEY,TARGETS,day_files,network,read_csv,save_json,sha256
from ..fusion_experiments import aligned_prediction
from ..grin.data import key_positions,window_starts
from ..grin.runner import blend_windows,comparison
from ..metrics import aggregate,score,validate_predictions
from ..pipeline import truth_rows,panel_families
from ..traffic_wave.fit import ordered_key_hash
from ..traffic_wave.runner import export_errors
from .data import Windows,training_statistics,physical_cache,ramp_files,input_day
from .model import WaveNeural
from .physics import tensors,losses,diagnostics
from .reference import CONTROL,METHODS,sources,control_check
from .log import append_log
from .cache import NeuralCache,save_checkpoint,save_history

PROJECT=Path(__file__).resolve().parents[2]


def variant(method,cfg):
    if method not in METHODS[1:]:
        raise ValueError('Unknown neural ablation')
    return dict(use_wave=method!='Neural_only',fd=cfg['lambda_fd'] if method in METHODS[3:] else 0.,
                lwr=cfg['lambda_lwr'] if method==METHODS[4] else 0.)


def validate(cfg):
    if cfg['panel']!='D12_I5_N' or cfg['seed']!=20261016 or cfg['sample_per_day']!=1000:
        raise ValueError('This first experiment fixes I5 and the original target pool')
    window_starts(cfg['window'],cfg['stride'])
    for name in ['batch_size','threads','epochs','steps_per_epoch','patience','hidden','layers','warmup_epochs']:
        if not isinstance(cfg[name],int) or cfg[name]<1:
            raise ValueError(name+' must be a positive integer')
    for name in ['learning_rate','lambda_fd','lambda_lwr']:
        if not math.isfinite(cfg[name]) or cfg[name]<=0:
            raise ValueError(name+' must be finite and positive')
    if cfg['layers']!=2 or cfg['hidden']!=32:
        raise ValueError('This controlled experiment fixes 2 layers and hidden=32')
    expected=dict(fit_start='2030-06-01',fit_end='2030-12-31',eval_start='2031-01-01',eval_end='2031-01-31')
    if any(cfg[k]!=v for k,v in expected.items()):
        raise ValueError('Fixed June-December / January split differs')


def seed(cfg):
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])


def weights_hash(model):
    result=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        result.update(name.encode());result.update(value.cpu().numpy().tobytes())
    return result.hexdigest()


def memory(device):
    if device.type!='cuda':return dict(measured=False,device='cpu')
    return dict(measured=True,device=torch.cuda.get_device_name(device),total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                note='Process allocator peak across all four networks; not whole-device usage')


@torch.inference_mode()
def predict(model,dataset,expected,nodes,cfg,device):
    model.eval();outputs=[];starts=window_starts(cfg['window'],cfg['stride'])
    for date in dataset.dates:
        day=dataset.day(date);parts=[]
        for first in range(0,len(starts),cfg['batch_size']):
            local=starts[first:first+cfg['batch_size']]
            batch={k:torch.from_numpy(np.stack([day[k][s:s+cfg['window']] for s in local])).to(device)
                   for k in ['features','prior','x','mask']}
            values=model(batch['features'],batch['prior'],batch['x'],batch['mask'])
            parts.append(values.cpu().numpy())
        values=blend_windows(np.concatenate(parts),starts)
        target=expected.loc[expected.timestamp.str.startswith(date)].copy()
        rr,cc=key_positions(target,nodes)
        selected=values[rr,cc].copy();selected[:,1]*=nodes.lanes.to_numpy()[cc]
        target[TARGETS]=selected;outputs.append(target)
    result=pd.concat(outputs,ignore_index=True)
    validate_predictions(expected,result)
    return result


def train_model(cache,method,cfg,records_train,records_eval,source,stats,physical_meta,means,attributes,adjacency,physical,
                expected,truth,net,family,nodes,train_keys,dev_truth_hashes,device):
    setting=variant(method,cfg)
    architecture={k:cfg[k] for k in ['window','stride','hidden','layers','batch_size','epochs','steps_per_epoch','patience',
                    'seed','learning_rate','threads','warmup_epochs']}
    identity=dict(kind='wave_neural_model',schema='wave_neural_ablation_v1',panel=cfg['panel'],method=method,setting=setting,
        architecture=architecture,training_input_keys=[r['key'] for r in records_train],evaluation_input_keys=[r['key'] for r in records_eval],
        training_grid_key=source['grid_train']['key'],evaluation_grid_key=source['grid_eval']['key'],
        ordered_training_keys_sha256=ordered_key_hash(train_keys),statistics_key=stats['key'],physical_key=physical_meta['key'],
        evaluation_keys_sha256=ordered_key_hash(expected),development_truth=dev_truth_hashes,
        initialization='same seed and architecture for all four; zero residual head',selection='highest official January S_state, patience early stopping',
        optimizer='AdamW weight_decay=1e-5; grad_clip=5',extra_observation_supervision=False,artificial_masking=False,
        backend=dict(device=str(device),name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',tf32=False),
        packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow']},
        shared_implementation=snapshot([PROJECT/'task1/grin/runner.py',PROJECT/'task1/traffic_wave/runner.py',PROJECT/'task1/data.py']),
        implementation=snapshot(list(Path(__file__).parent.glob('*.py')))+[dict(path='official_metrics',sha256=sha256(PROJECT/'task1/metrics.py'))])
    train_set=Windows(records_train,source['grid_train']['metadata'],means,attributes,setting['use_wave'],True,cfg['window'],cfg['stride'])
    eval_set=Windows(records_eval,source['grid_eval']['metadata'],means,attributes,setting['use_wave'],False,cfg['window'],cfg['stride'])
    def build(destination):
        seed(cfg);model=WaveNeural(adjacency,cfg['hidden'],cfg['layers']).to(device)
        initial_hash=weights_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-5)
        best,best_epoch=-float('inf'),0;history=[]
        for epoch in range(1,cfg['epochs']+1):
            began=time.perf_counter();model.train();totals=[]
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            loader=DataLoader(train_set,batch_size=cfg['batch_size'],shuffle=True,generator=generator,num_workers=0)
            warm=min(1.,epoch/cfg['warmup_epochs'])
            fd_weight,lwr_weight=setting['fd']*warm,setting['lwr']*warm
            for step,batch in enumerate(loader,1):
                batch={k:v.to(device) for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                values=model(batch['features'],batch['prior'],batch['x'],batch['mask'])
                data,fd,lwr,count=losses(values,batch['y'],batch['supervise'],batch['mask'],batch['ramps'],batch['ramp_valid'],physical)
                loss=data+fd_weight*fd+lwr_weight*lwr
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural/physics loss in '+method)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                totals.append([float(loss.detach()),float(data.detach()),float(fd.detach()),float(lwr.detach()),int(count)])
                if step%20==0:print(f'[{method}] epoch={epoch}; step={step}/{min(len(loader),cfg["steps_per_epoch"])}',flush=True)
                if step>=cfg['steps_per_epoch']:break
            prediction=predict(model,eval_set,expected,nodes,cfg,device)
            detail,_=score(prediction,truth,net,family,method)
            value=float(aggregate(pd.DataFrame(detail)).S_state.iloc[0])
            row=dict(epoch=epoch,steps=len(totals),loss=float(np.mean(totals,axis=0)[0]),data_loss=float(np.mean(totals,axis=0)[1]),
                     fd_proxy_loss=float(np.mean(totals,axis=0)[2]),lwr_proxy_loss=float(np.mean(totals,axis=0)[3]),
                     conservation_transitions_seen=int(np.sum(totals,axis=0)[4]),fd_weight=fd_weight,lwr_weight=lwr_weight,
                     dev_S_state=value,seconds=time.perf_counter()-began)
            history.append(row);save_history(destination/'training_history.csv',pd.DataFrame(history))
            if value>best:
                best,best_epoch=value,epoch
                save_checkpoint(destination/'best.pt',dict(state=model.state_dict(),best_epoch=epoch,dev_S_state=value,initial_weights_sha256=initial_hash))
            print(f'[{method}] epoch={epoch}; loss={row["loss"]:.6f}; dev={value:.6f}; best_epoch={best_epoch}; seconds={row["seconds"]:.1f}',flush=True)
            if epoch-best_epoch>=cfg['patience']:break
        result=dict(method=method,setting=setting,best_epoch=best_epoch,best_dev_S_state=best,epochs_run=len(history),
                    initial_weights_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),training_rows=len(train_keys),
                    training_steps=sum(r['steps'] for r in history),checkpoint_selection='January official S_state; February untouched')
        save_json(destination/'selection.json',result)
        del model,optimizer;train_set.day.cache_clear();eval_set.day.cache_clear()
        return result
    folder,manifest,hit=cache.get_or_build(identity,build)
    seed(cfg);model=WaveNeural(adjacency,cfg['hidden'],cfg['layers']).to(device)
    if weights_hash(model)!=manifest['metadata']['initial_weights_sha256']:
        raise ValueError('Initial network weights differ between ablations/cache')
    checkpoint=torch.load(folder/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state'])
    prediction=predict(model,eval_set,expected,nodes,cfg,device)
    actual,_=score(prediction,truth,net,family,method)
    selected=float(aggregate(pd.DataFrame(actual)).S_state.iloc[0])
    if abs(selected-manifest['metadata']['best_dev_S_state'])>1e-9:
        raise ValueError('Loaded checkpoint does not reproduce its selection score')
    del model;train_set.day.cache_clear();eval_set.day.cache_clear()
    return prediction,dict(manifest['metadata'],key=manifest['key'],folder=str(folder),hit=hit)


def physics_report(predictions,expected,records,nodes,physical,output):
    rows=[];node_count=len(nodes)
    for method,prediction in predictions.items():
        aligned=aligned_prediction(expected,prediction);sums=None
        for record in records:
            date=record['date']
            with np.load(Path(record['grid_folder'])/(date+'.npz'),allow_pickle=False) as a:x,mask=a['x'],a['mask']
            with np.load(Path(record['folder'])/'inputs.npz',allow_pickle=False) as a:
                interp,ramps,valid=a['interp'],a['ramp_net'],a['ramp_valid']
            # Every method shares the SAME background at non-submission holes.
            dense=np.where(mask,x,interp).copy();active=np.zeros_like(mask)
            target=aligned.loc[aligned.timestamp.str.startswith(date)]
            rr,cc=key_positions(target,nodes);values=target[TARGETS].to_numpy(float,copy=True);values[:,1]/=nodes.lanes.to_numpy()[cc]
            dense[rr,cc]=values;active[rr,cc]=True
            part=diagnostics(torch.from_numpy(dense[None]),torch.from_numpy(active[None]),torch.from_numpy(ramps[None]),
                             torch.from_numpy(valid[None]),physical)
            sums=part if sums is None else {k:sums[k]+part[k] for k in sums}
        row=dict(method=method,**sums)
        row.update(fd_squared_excess_mean=sums['fd_squared_excess_sum']/max(sums['fd_cells'],1),
                   fd_violation_rate=sums['fd_violation_cells']/max(sums['fd_cells'],1),
                   conservation_mae_veh=sums['conservation_abs_residual_veh_sum']/max(sums['conservation_transitions'],1),
                   conservation_normalized_excess_mean=sums['conservation_normalized_excess_sum']/max(sums['conservation_transitions'],1))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'physics_proxy_diagnostics.csv',index=False)
    save_json(output/'physics_proxy_definition.json',dict(official_S_physics=False,density_speed_floor_kmh=1,
        background='same temporal input prior for all methods at non-target missing cells; observed data unchanged',
        active_cells='submission target cells; transitions require at least one active involved state',
        conservation='strict internal one-in-one-out midpoint-flux proxy; excludes unknown connectors/branches',
        node_count=node_count,conservation_nodes=int(physical['valid_nodes'].sum()),
        tolerance='FD 10%; conservation 5 vehicles + 5% of nominal five-minute capacity'))


def run(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new empty output directory')
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(panel='D12_I5_N',seed=20261016,sample_per_day=1000,fit_start='2030-06-01',fit_end='2030-12-31',
        eval_start='2031-01-01',eval_end='2031-01-31',window=48,stride=24,hidden=32,layers=2,epochs=2 if args.smoke else 20,
        steps_per_epoch=4 if args.smoke else 100,patience=5,learning_rate=.001,lambda_fd=.01,lambda_lwr=.01,warmup_epochs=3)
    validate(cfg);start=time.perf_counter();seed(cfg)
    torch.set_num_threads(cfg['threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run setup_wave_neural.ps1 or explicitly use CPU')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    print(f'[wave neural] I5; {device}; mode={"smoke" if args.smoke else "formal"}; five methods',flush=True)
    source=sources(cfg);nodes=pd.DataFrame(source['grid_train']['metadata']['nodes'])
    train_dates=source['grid_train']['metadata']['dates'][:14] if args.smoke else source['grid_train']['metadata']['dates']
    eval_dates=source['grid_eval']['metadata']['dates'][:3] if args.smoke else source['grid_eval']['metadata']['dates']
    train_keys=source['keys'].loc[source['keys'].timestamp.str[:10].isin(train_dates)].reset_index(drop=True)
    expected=source['expected'].loc[source['expected'].timestamp.str[:10].isin(eval_dates)].reset_index(drop=True)
    control=aligned_prediction(expected,source['control'].loc[source['control'].timestamp.str[:10].isin(eval_dates)])
    diagnostics_table=expected.merge(source['diagnostics'],on=KEY,validate='one_to_one')
    root=Path(cfg['release_root']);net=network(root,cfg['panel']);family=panel_families(root)[cfg['panel']]
    truth=truth_rows(root,cfg['panel'],expected)
    control_rows,_=score(control,truth,net,family,CONTROL)
    check=control_check(source,pd.DataFrame(control_rows),eval_dates,args.smoke)
    cache=NeuralCache(Path(cfg['cache_dir']))
    stats_folder,stats,stats_hit=training_statistics(cache,source['grid_train_folder'],source['grid_train'])
    physical_folder,pm,physical_hit=physical_cache(cache,root,cfg['panel'],nodes)
    with np.load(physical_folder/'structure.npz',allow_pickle=False) as a:physical_np={k:a[k] for k in a.files}
    physical_gpu=tensors(physical_np,device)
    means=np.load(stats_folder/'means.npy',allow_pickle=False);attributes=np.load(physical_folder/'attributes.npy',allow_pickle=False)
    ramps=ramp_files(root,cfg['panel']);attachments=read_csv(root/'corridors'/cfg['panel']/'network/ramp_attachment_map.csv')
    output.mkdir(parents=True,exist_ok=True)
    cfg.update(actual_fit_dates=train_dates,actual_eval_dates=eval_dates)
    save_json(output/'run_config.json',cfg)
    save_json(output/'reference_check.json',dict(passed=True,check=check,subset=args.smoke,current_best_source=cfg['wave_run']))
    save_json(output/'sample_check.json',dict(passed=True,formal_pool_rows=214000,used_rows=len(train_keys),
        exact_ordered_keys_equal_to_baseline_and_grin=True,ordered_keys_sha256=ordered_key_hash(train_keys),
        artificial_masking=False,extra_visible_supervision=False))
    save_json(output/'physical_coverage.json',pm['metadata'])
    usage=dict(original_grid_cache_reused=True,original_wave_cache_reused=True,original_models_retrained=False,
        statistics=dict(key=stats['key'],hit=stats_hit),physical_structure=dict(key=pm['key'],hit=physical_hit),
        training_days=[],evaluation_days=[],models=[])
    for name,dates in [('training',train_dates),('evaluation',eval_dates)]:
        for i,date in enumerate(dates,1):
            wave_folder,wave_manifest=source['wave_records'][date]
            record=input_day(cache,date,source['grid_train_folder' if name=='training' else 'grid_eval_folder'],
                source['grid_train' if name=='training' else 'grid_eval'],wave_folder,wave_manifest,stats_folder,stats,pm,ramps[date],attachments,nodes)
            usage[name+'_days'].append(record)
            if i%20==0:print(f'[neural inputs] {name} {i}/{len(dates)}',flush=True)
    save_json(output/'cache_usage.json',usage)
    save_json(output/'environment.json',dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow']},
        device=str(device),cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,source_files=snapshot(Path(__file__).parent.glob('*.py')),
        metric_sha256=sha256(PROJECT/'task1/metrics.py')))
    if args.action=='prepare':
        save_json(output/'log_update.json',dict(status='skipped_prepare'))
        print(f'Preflight passed: {len(train_keys)} training, {len(expected)} evaluation targets; {len(nodes)} nodes; no model trained. {output}',flush=True)
        return
    full=day_files(root,cfg['panel'],'train',False);truth_hashes=snapshot(full[d] for d in eval_dates)
    adjacency=np.load(source['grid_train_folder']/'adjacency.npy',allow_pickle=False)
    predictions={CONTROL:control};all_rows=export_errors(output,CONTROL,control,truth,net,family,diagnostics_table)
    control.to_parquet(output/(CONTROL+'_predictions.parquet'),index=False)
    for method in METHODS[1:]:
        prediction,record=train_model(cache,method,cfg,usage['training_days'],usage['evaluation_days'],source,stats,pm,means,attributes,
            adjacency,physical_gpu,expected,truth,net,family,nodes,train_keys,truth_hashes,device)
        predictions[method]=prediction;usage['models'].append(record);save_json(output/'cache_usage.json',usage)
        prediction.to_parquet(output/(method+'_predictions.parquet'),index=False)
        all_rows+=export_errors(output,method,prediction,truth,net,family,diagnostics_table)
        pd.read_csv(Path(record['folder'])/'training_history.csv').to_csv(output/(method+'_training_history.csv'),index=False)
        save_json(output/(method+'_selection.json'),{k:v for k,v in record.items() if k not in ['folder']})
    initial_hashes={r['initial_weights_sha256'] for r in usage['models']}
    if len(initial_hashes)!=1:raise ValueError('Four networks did not start from identical weights')
    detail=pd.DataFrame(all_rows);table=comparison(detail);control_score=float(table.loc[table.method.eq(CONTROL),'S_state'].iloc[0])
    table['delta_S_state_vs_current_best']=table.S_state-control_score
    detail.to_csv(output/'metrics_by_panel_regime.csv',index=False);aggregate(detail).to_csv(output/'metrics_summary.csv',index=False)
    table.to_csv(output/'comparison_methods.csv',index=False)
    physics_report(predictions,expected,usage['evaluation_days'],nodes,tensors(physical_np,'cpu'),output)
    save_json(output/'memory.json',memory(device))
    integrity=dict(passed=True,all_five_methods_same_targets=True,all_four_initial_weights_equal=True,
        original_control_predictions_reused=True,new_models_apply_to_all_gaps=True,original_models_retrained=False,
        input_feature_count=31,neural_model_count=4,physical_proxy_only=True,conservation_nodes=pm['metadata']['conservation_nodes'])
    save_json(output/'integrity_check.json',integrity)
    save_json(output/'source_manifest.json',snapshot(source['provenance']+[ramp for date,ramp in ramps.items() if date in train_dates+eval_dates]+
        [stats_folder/'manifest.json',physical_folder/'manifest.json']+[Path(r['folder'])/'manifest.json' for r in usage['training_days']+usage['evaluation_days']+usage['models']]))
    save_json(output/'COMPLETE.json',dict(experiment='wave_neural',panel=cfg['panel'],smoke=args.smoke,methods=METHODS,
        training_rows=len(train_keys),evaluation_rows=len(expected),reference_passed=True,pairing_passed=True,integrity_passed=True,
        all_regimes_present=bool(table.all_regimes_present.all()),seconds=time.perf_counter()-start))
    try:status=append_log(output,PROJECT/'实验日志.txt')
    except (ValueError,OSError) as error:
        save_json(output/'log_update.json',dict(status='failed',error=str(error)))
        raise RuntimeError('Evaluation complete; retry log without training') from error
    print(table.to_string(index=False),flush=True)
    print(f'Log status: {status["status"]}. Wave neural complete: {output}',flush=True)
