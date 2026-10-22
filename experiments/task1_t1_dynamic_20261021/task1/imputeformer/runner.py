from __future__ import annotations

import argparse
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

from ..cache import FeatureCache, snapshot
from ..data import KEY, TARGETS, day_files, network, save_json, sha256
from ..fusion_experiments import aligned_prediction
from ..metrics import aggregate, score
from ..pipeline import panel_families, truth_rows
from ..grin.data import GraphWindows, grid_cache
from ..grin.runner import comparison, predict, validate_protocol as grin_protocol
from ..grin.hybrid import check_scores, load_baseline, require_compatible_grin, switch_predictions
from .model import ImputeFormer, UPSTREAM, UPSTREAM_COMMIT, reconstruction_loss
from .log import append_log

PROJECT = Path(__file__).resolve().parents[2]
METHOD, BASELINE, GRIN, OLD_HYBRID, HYBRID = (
    'ImputeFormer', 'Fusion_gap60mean', 'GRIN', 'Baseline_GRIN_gap60', 'Baseline_ImputeFormer_gap60')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_protocol(cfg):
    grin_protocol(dict(cfg, aux_weight=0))
    if cfg['panel'] != 'D12_I5_N':
        raise ValueError('This first ImputeFormer experiment is fixed to D12_I5_N')
    if not math.isfinite(cfg['learning_rate']) or cfg['learning_rate'] <= 0:
        raise ValueError('Learning rate must be finite and positive')
    for key in ['hidden','embedding_dim','heads','layers','rank','feed_forward','batch_size','epochs','steps_per_epoch','patience','threads']:
        if cfg[key] < 1: raise ValueError(f'{key} must be positive')
    if cfg['embedding_dim'] >= cfg['hidden'] or cfg['hidden'] % cfg['heads'] or cfg['rank'] > cfg['window']:
        raise ValueError('Invalid embedding/head/rank configuration')
    if not math.isfinite(cfg['drop_observed']) or not math.isfinite(cfg['dropout']):
        raise ValueError('Masking/dropout must be finite')
    if not 0 <= cfg['dropout'] < 1:
        raise ValueError('Dropout must be in [0,1)')
    if (cfg.get('max_fit_days') or 214) > 214 or (cfg.get('max_eval_days') or 31) > 31:
        raise ValueError('Day limit exceeds the fixed date interval')


def model_for(cfg, nodes):
    return ImputeFormer(nodes, **{key:cfg[key] for key in [
        'window','hidden','embedding_dim','heads','layers','rank','feed_forward','dropout']})


def check_samples(cfg, folder, source=None):
    if cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        return dict(status='skipped_smoke', passed=False, reason='Budgeted 7-day / 300-per-day sample is not the formal sample')
    source = Path(cfg['baseline_sample_table']) if cfg.get('baseline_sample_table') else source
    if source is None or not source.is_file():
        raise FileNotFoundError('Formal run requires the original baseline training target table')
    original = pd.read_parquet(source, columns=['__'+k for k in KEY]).rename(columns={'__'+k:k for k in KEY})
    current = pd.read_parquet(folder/'training_sample_keys.parquet')
    if not original.reset_index(drop=True).equals(current[KEY].reset_index(drop=True)) or len(current) != 214000:
        raise ValueError('ImputeFormer training sample IDs differ from frozen baseline')
    return dict(passed=True, same_ordered_training_targets=True, rows=len(current), baseline_table=str(source))


def load_controls(cfg, training, evaluation, expected, truth, net, family):
    baseline = load_baseline(cfg['baseline_run'], cfg['panel'])
    source = Path(cfg['grin_run']); old_cfg = read(source/'run_config.json')
    require_compatible_grin(source, old_cfg)
    for key in ['panel','release_root','fit_start','fit_end','eval_start','eval_end','sample_per_day','seed','max_fit_days','max_eval_days']:
        if cfg[key] != old_cfg[key]: raise ValueError(f'GRIN control data protocol differs: {key}')
    old_usage = read(source/'cache_usage.json')
    if [training['key'],evaluation['key']] != [old_usage['training_key'],old_usage['evaluation_key']]:
        raise ValueError('ImputeFormer/GRIN cached data and normalization identities differ')
    hybrid = Path(cfg['hybrid_run']); complete = read(hybrid/'COMPLETE.json'); hybrid_cfg = read(hybrid/'run_config.json')
    if complete.get('experiment') != 'grin_hybrid' or not complete.get('reference_passed') or hybrid_cfg.get('smoke'):
        raise ValueError('Existing GRIN hybrid control must be a completed formal run')
    if Path(hybrid_cfg['baseline_run']).resolve() != Path(cfg['baseline_run']).resolve():
        raise ValueError('Hybrid baseline source differs')
    if Path(read(hybrid/'reuse.json')['grin_source']).resolve() != source.resolve():
        raise ValueError('Hybrid GRIN source differs')
    diagnostic = expected.merge(baseline['diagnostics'], on=KEY, validate='one_to_one')
    if len(diagnostic) != len(baseline['diagnostics']):
        raise ValueError('Control target coverage differs')
    predictions = {
        BASELINE: aligned_prediction(expected, baseline['baseline']),
        GRIN: aligned_prediction(expected, pd.read_parquet(source/'GRIN_predictions.parquet')),
        OLD_HYBRID: aligned_prediction(expected, pd.read_parquet(hybrid/'combined_predictions.parquet')),
    }
    reconstructed = switch_predictions(expected, predictions[BASELINE], predictions[GRIN], diagnostic.gap_minutes)
    if not np.array_equal(reconstructed[TARGETS].to_numpy(), predictions[OLD_HYBRID][TARGETS].to_numpy()):
        raise ValueError('Saved GRIN hybrid does not match the fixed gap rule')
    references = pd.read_csv(hybrid/'metrics_by_panel_regime.csv')
    checks, rows = [], []
    for method, pred in predictions.items():
        detail, _ = score(pred, truth, net, family, method)
        checks.append(check_scores(references, pd.DataFrame(detail), method)); rows += detail
    if not all(check['passed'] for check in checks):
        raise ValueError('Existing baseline/GRIN/hybrid reference scores failed to reproduce')
    provenance = baseline['provenance'] + [source/name for name in [
        'GRIN_predictions.parquet','run_config.json','environment.json','COMPLETE.json']]
    provenance += [hybrid/name for name in ['combined_predictions.parquet','metrics_by_panel_regime.csv','COMPLETE.json','run_config.json','reuse.json']]
    return dict(predictions=predictions, rows=rows, checks=checks, diagnostics=diagnostic,
                sample_table=baseline['training_table'], provenance=provenance)


def write_errors(output, method, pred, truth, net, family, gaps):
    rows, cells = score(pred, truth, net, family, method)
    # score preserves prediction metadata; attach one authoritative observable gap.
    cells = cells.drop(columns=['gap','selected_model'], errors='ignore').merge(
        gaps[KEY+['gap']], on=KEY, validate='one_to_one')
    for label, grouping in [('daily',['date','mask_regime']),('gap',['gap']),('congestion',['congested'])]:
        cells.groupby(grouping, as_index=False).agg(n=('speed_sq_error','size'),
            speed_mse=('speed_sq_error','mean'), flow_lane_mse=('flow_lane_sq_error','mean')).to_csv(
                output/f'{method}_{label}_errors.csv', index=False)
    return rows


def memory(device):
    if device.type != 'cuda': return dict(device='cpu', measured=False)
    return dict(device=torch.cuda.get_device_name(device), measured=True,
        total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        note='PyTorch process allocator peaks; not total device usage by all applications')


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError('Choose a new, empty result directory')
    cfg = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    implementation = {str(p.relative_to(PROJECT)):sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    implementation.update({str(p.relative_to(PROJECT)):sha256(p) for p in [
        PROJECT/'task1/grin/data.py',PROJECT/'task1/grin/runner.py',PROJECT/'task1/grin/hybrid.py',PROJECT/'task1/metrics.py']})
    checkpoint = None
    if args.action == 'evaluate':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if checkpoint.get('implementation_sha256') != implementation:
            raise ValueError('Checkpoint implementation differs from current model/shared inference/scoring')
        cfg = dict(checkpoint['config'], action='evaluate', checkpoint=str(args.checkpoint),
            output=str(output), device=args.device, threads=args.threads, batch_size=args.batch_size, no_log=args.no_log)
    validate_protocol(cfg)
    smoke = cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['threads']); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    if cfg['device']=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; check setup_imputeformer.ps1')
    device = torch.device('cuda' if cfg['device']=='cuda' or (cfg['device']=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    print(f'[ImputeFormer] device={device}; torch={torch.__version__}; hidden={cfg["hidden"]}; layers={cfg["layers"]}; batch={cfg["batch_size"]}', flush=True)
    root = Path(cfg['release_root']); cache=FeatureCache(Path(cfg['cache_dir']))
    train_folder, training, train_hit = grid_cache(cache,root,cfg,'training')
    eval_folder, evaluation, eval_hit = grid_cache(cache,root,cfg,'evaluation',parent=training)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'run_config.json',cfg)
    save_json(output/'cache_usage.json',dict(training_key=training['key'],training_hit=train_hit,
        evaluation_key=evaluation['key'],evaluation_hit=eval_hit,shared_with='GRIN daily grid v1',
        evaluation_cache_contains_truth=False,training_scaler=training['metadata']['scaler'],
        baseline_retrained=False,grin_retrained=False))
    save_json(output/'environment.json',dict(python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow']},
        device=str(device),cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        source_sha256=implementation,upstream=UPSTREAM,upstream_commit=UPSTREAM_COMMIT,
        adaptation='Compact core structure; speed+flow+mask+time input, score-scaled MSE; no Fourier regularizer'))
    # Formal sample verification is mandatory, including prepare-only runs.
    sample_table=None
    if not smoke:
        records=read(Path(cfg['baseline_run'])/'cache_usage.json')
        record=next(r for r in records if r['panel']==cfg['panel'])
        base_cfg=read(Path(cfg['baseline_run'])/'run_config.json')
        sample_table=Path(base_cfg['cache_dir'])/record['training_key']/'training.parquet'
    sample_check=check_samples(cfg,train_folder,sample_table)
    save_json(output/'sample_check.json',sample_check)
    expected=pd.read_parquet(eval_folder/'expected_keys.parquet')
    truth,net=truth_rows(root,cfg['panel'],expected),network(root,cfg['panel'])
    family=panel_families(root)[cfg['panel']]
    controls=None if smoke else load_controls(cfg,training,evaluation,expected,truth,net,family)
    save_json(output/'reference_check.json',dict(passed=bool(controls),
        status='skipped_smoke' if smoke else 'all_three_saved_controls_reproduced',checks=controls['checks'] if controls else []))
    if args.action=='prepare':
        if controls:comparison(pd.DataFrame(controls['rows'])).to_csv(output/'comparison_controls.csv',index=False)
        save_json(output/'log_update.json',dict(status='skipped_prepare'))
        print(f'ImputeFormer grids, samples and available controls checked (no training): {output}',flush=True);return
    model=model_for(cfg,len(training['metadata']['nodes'])).to(device)
    if checkpoint:
        if checkpoint['training_key']!=training['key'] or checkpoint['evaluation_key']!=evaluation['key']:
            raise ValueError('Checkpoint grid/normalization identity differs')
        model.load_state_dict(checkpoint['model'])
    best_score,best_epoch=(-float('inf'),0) if checkpoint is None else (checkpoint['dev_S_state'],checkpoint['best_epoch'])
    if checkpoint is None:
        dataset=GraphWindows(train_folder,training,cfg['window'],cfg['stride'])
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-5)
        channel_scale=torch.tensor(training['metadata']['scaler']['std'],device=device)/torch.tensor([25.,600.],device=device)
        history=[]
        for epoch in range(1,cfg['epochs']+1):
            began=time.perf_counter();model.train();losses=[];original_losses=[];random_losses=[];original_counts=[];random_counts=[]
            loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,
                generator=torch.Generator().manual_seed(cfg['seed']+epoch),num_workers=0)
            for step,batch in enumerate(loader,1):
                x,mask,u,y,supervised=[part.to(device) for part in batch]
                removed=(torch.rand((*mask.shape[:-1],1),device=device)<cfg['drop_observed']) & mask
                target=torch.where(removed,x,y);active=supervised|removed
                optimizer.zero_grad(set_to_none=True)
                prediction,_=model(x,mask & ~removed,u)
                loss=reconstruction_loss(prediction,target,active,channel_scale)
                if not torch.isfinite(loss): raise RuntimeError('Non-finite ImputeFormer loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                losses.append(float(loss.detach()))
                with torch.no_grad():
                    original_losses.append(float(reconstruction_loss(prediction,y,supervised,channel_scale)))
                    random_losses.append(float(reconstruction_loss(prediction,x,removed,channel_scale)))
                    original_counts.append(int(supervised.sum()));random_counts.append(int(removed.sum()))
                if step%20==0: print(f'[ImputeFormer] epoch={epoch}, step={step}/{min(len(loader),cfg["steps_per_epoch"])}',flush=True)
                if step>=cfg['steps_per_epoch']:break
            train_seconds=time.perf_counter()-began
            prediction=predict(model,eval_folder,evaluation,cfg,device)
            rows,_=score(prediction,truth,net,family,METHOD)
            value=float(aggregate(pd.DataFrame(rows)).S_state.iloc[0])
            history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),
                original_hidden_loss=float(np.mean(original_losses)),artificial_mask_loss=float(np.mean(random_losses)),
                original_hidden_channel_labels=sum(original_counts),artificial_mask_channel_labels=sum(random_counts),
                dev_S_state=value,training_seconds=train_seconds,seconds=time.perf_counter()-began,**memory(device)))
            pd.DataFrame(history).to_csv(output/'training_history.csv',index=False)
            if value>best_score:
                best_score,best_epoch=value,epoch
                temporary=output/'best.tmp'
                torch.save(dict(model=model.state_dict(),config=cfg,training_key=training['key'],evaluation_key=evaluation['key'],
                    implementation_sha256=implementation,best_epoch=epoch,dev_S_state=value),temporary)
                temporary.replace(output/'best.pt')
            print(f'[ImputeFormer] epoch={epoch}, loss={history[-1]["training_loss"]:.6f}, dev={value:.6f}, best_epoch={best_epoch}, seconds={history[-1]["seconds"]:.1f}',flush=True)
            if epoch-best_epoch>=cfg['patience']:break
        checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['model'])
    prediction=predict(model,eval_folder,evaluation,cfg,device)
    prediction.to_parquet(output/'ImputeFormer_predictions.parquet',index=False)
    predictions={METHOD:prediction};gaps=prediction[KEY+['gap']]
    if controls:
        predictions.update(controls['predictions'])
        combined=switch_predictions(expected,predictions[BASELINE],prediction,controls['diagnostics'].gap_minutes)
        combined['selected_model']=combined.selected_model.replace({'GRIN':METHOD})
        combined.to_parquet(output/'combined_predictions.parquet',index=False)
        combined.groupby(['mask_regime','gap','selected_model'],as_index=False).size().rename(columns={'size':'n'}).to_csv(output/'gate_counts.csv',index=False)
        predictions[HYBRID]=combined
    rows=[]
    for method,pred in predictions.items(): rows+=write_errors(output,method,pred,truth,net,family,gaps)
    detail=pd.DataFrame(rows);table=comparison(detail)
    for method,label in [(GRIN,'GRIN'),(OLD_HYBRID,'current_hybrid')]:
        reference=table.loc[table.method.eq(method)]
        if not reference.empty:table[f'delta_S_state_vs_{label}']=table.S_state-float(reference.S_state.iloc[0])
    detail.to_csv(output/'metrics_by_panel_regime.csv',index=False)
    aggregate(detail).to_csv(output/'metrics_summary.csv',index=False)
    table.to_csv(output/'comparison_models.csv',index=False)
    save_json(output/'selection.json',dict(best_epoch=best_epoch,best_dev_S_state=best_score,
        checkpoint_selection='Standalone ImputeFormer January S_state, not hybrid score; no January gradients; February untouched',
        training_rows=training['metadata']['sampled_training_rows'],nodes=model.nodes,parameters=sum(p.numel() for p in model.parameters()),
        training_objective=f'Score-scaled MSE with original sampled hidden targets + {cfg["drop_observed"]:.2%} artificial point masking; no auxiliary or Fourier loss',
        fusion_rule='Finite bilateral speed bracketing gap >60min: ImputeFormer; otherwise frozen baseline; missing side stays baseline'))
    save_json(output/'memory.json',memory(device))
    sources=[train_folder/'manifest.json',eval_folder/'manifest.json']+list(Path(__file__).parent.glob('*.py'))
    sources += controls['provenance'] if controls else []
    truth_files=day_files(root,cfg['panel'],'train',False)
    sources += [truth_files[day] for day in evaluation['metadata']['dates']]
    save_json(output/'source_manifest.json',snapshot(sources))
    save_json(output/'COMPLETE.json',dict(experiment='imputeformer_dev',panel=cfg['panel'],smoke=smoke,
        methods=list(predictions),evaluation_rows=len(prediction),best_epoch=best_epoch,
        pairing_passed=bool(sample_check['passed']),reference_passed=bool(controls)))
    try: status=append_log(output,PROJECT/'实验日志.txt')
    except (OSError,ValueError) as error:
        save_json(output/'log_update.json',dict(status='failed',error=str(error)))
        raise RuntimeError('ImputeFormer completed; retry log action without training') from error
    print(table.to_string(index=False),flush=True)
    print(f'Memory: {memory(device)}',flush=True)
    print(f'Log status: {status["status"]}. ImputeFormer complete: {output}',flush=True)
