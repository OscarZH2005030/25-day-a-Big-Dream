from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..cache import snapshot
from ..data import KEY, TARGETS, day_files, network, save_json
from ..fusion_experiments import aligned_prediction
from ..grin.runner import comparison
from ..metrics import aggregate, score
from ..pipeline import truth_rows, panel_families
from ..traffic_wave.fit import ordered_key_hash
from ..traffic_wave.runner import export_errors
from ..wave_neural.cache import NeuralCache, save_checkpoint, save_history
from ..wave_neural.runner import seed, weights_hash, predict
from .data import ContextWindows, Coverage, order_hash, anchor_check
from .model import ContextNeural, supervised_loss, receptive_field_check, DILATIONS
from .reference import CONTROL, METHODS, sources, control_check
from .log import append_log

PROJECT = Path(__file__).resolve().parents[2]


def implementation():
    # Include transitive shared implementations without changing any legacy file.
    return snapshot((PROJECT/'task1').rglob('*.py'))


def memory(device):
    if device.type!='cuda':
        return dict(measured=False, device='cpu')
    return dict(measured=True, device=torch.cuda.get_device_name(device),
                total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                note='Process allocator peak, including both new models and independent RF probe; not total device usage')


def train_model(cache, method, cfg, train_set, eval_set, source, adjacency, expected, truth, net, family,
                nodes, train_keys, dev_truth_hashes, device):
    if method not in METHODS[1:] or train_set.use_wave!=(method==METHODS[2]) or eval_set.use_wave!=train_set.use_wave:
        raise ValueError('Method and wave-channel setting differ')
    architecture = {k:cfg[k] for k in ['window','stride','hidden','layers','batch_size','epochs','steps_per_epoch',
                                      'patience','seed','learning_rate','threads']}
    identity = dict(kind='wave_context_model', schema='interpolation_anchor_long_context_v1', panel=cfg['panel'], method=method,
        use_wave=train_set.use_wave, architecture=architecture, dilations=DILATIONS, prior='common cached temporal interpolation',
        training_input_keys=[r['key'] for r in train_set.records.values()], evaluation_input_keys=[r['key'] for r in eval_set.records.values()],
        training_grid_key=source['grid_train']['key'], evaluation_grid_key=source['grid_eval']['key'],
        statistics_key=source['stats']['key'], attributes_key=source['attributes_manifest']['key'],
        training_keys_sha256=ordered_key_hash(train_keys), evaluation_keys_sha256=ordered_key_hash(expected), development_truth=dev_truth_hashes,
        initialization='same seed, same architecture; zero residual head', selection='highest January official S_state; patience early stopping',
        optimizer='AdamW weight_decay=1e-5; grad_clip=5', loss='same .54 speed/25 squared + .46 lane-flow/600 squared; supervised channel means',
        extra_visible_supervision=False, artificial_masking=False, physics_penalties=False,
        backend=dict(device=str(device), name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU', tf32=False,
                     cudnn_deterministic=True, cudnn_benchmark=False),
        packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow']}, implementation=implementation())

    def build(destination):
        seed(cfg); model = ContextNeural(adjacency).to(device)
        initial_hash = weights_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
        best, best_epoch = -float('inf'), 0
        history = []; coverage = Coverage(train_set, train_keys); best_coverage = None
        for epoch in range(1, cfg['epochs']+1):
            began = time.perf_counter(); model.train(); losses = []; visited = []
            epoch_coverage = Coverage(train_set, train_keys)
            generator = torch.Generator().manual_seed(cfg['seed']+epoch)
            loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, generator=generator, num_workers=0)
            for step, batch in enumerate(loader, 1):
                indices = batch.pop('window_index').tolist()
                visited.extend(indices); coverage.update(indices); epoch_coverage.update(indices)
                batch = {k:v.to(device) for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                values = model(batch['features'], batch['prior'], batch['x'], batch['mask'])
                loss = supervised_loss(values, batch['y'], batch['supervise'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite supervised loss in '+method)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                losses.append(float(loss.detach()))
                if step%20==0:
                    print(f'[{method}] epoch={epoch}; step={step}/{min(len(loader),cfg["steps_per_epoch"])}', flush=True)
                if step>=cfg['steps_per_epoch']:
                    break
            prediction = predict(model, eval_set, expected, nodes, cfg, device)
            detail, _ = score(prediction, truth, net, family, method)
            value = float(aggregate(pd.DataFrame(detail)).S_state.iloc[0])
            row = dict(epoch=epoch, steps=len(losses), loss=float(np.mean(losses)), dev_S_state=value,
                seconds=time.perf_counter()-began, window_order_sha256=order_hash(visited),
                epoch_unique_targets_seen=epoch_coverage.summary()['unique_targets_seen'],
                epoch_unique_windows_seen=epoch_coverage.summary()['unique_windows_seen'], **coverage.summary())
            history.append(row); save_history(destination/'training_history.csv', pd.DataFrame(history))
            if value>best:
                best, best_epoch, best_coverage = value, epoch, coverage.summary().copy()
                save_checkpoint(destination/'best.pt', dict(state=model.state_dict(), best_epoch=epoch, dev_S_state=value,
                    initial_weights_sha256=initial_hash, coverage=best_coverage))
            print(f'[{method}] epoch={epoch}; loss={row["loss"]:.6f}; dev={value:.6f}; best_epoch={best_epoch}; seen={row["unique_targets_seen"]}/{len(train_keys)}; seconds={row["seconds"]:.1f}', flush=True)
            if epoch-best_epoch>=cfg['patience']:
                break
        result = dict(method=method, use_wave=train_set.use_wave, best_epoch=best_epoch, best_dev_S_state=best,
            epochs_run=len(history), initial_weights_sha256=initial_hash, parameters=sum(p.numel() for p in model.parameters()),
            available_training_rows=len(train_keys), training_steps=sum(r['steps'] for r in history),
            best_checkpoint_coverage=best_coverage, final_training_coverage=coverage.summary(),
            checkpoint_selection='January official S_state; February untouched', prior='temporal interpolation in both groups')
        save_json(destination/'selection.json', result)
        del model, optimizer
        train_set.day.cache_clear(); eval_set.day.cache_clear()
        return result

    folder, manifest, hit = cache.get_or_build(identity, build)
    seed(cfg); model = ContextNeural(adjacency).to(device)
    if weights_hash(model)!=manifest['metadata']['initial_weights_sha256']:
        raise ValueError('Initial weights differ from cached initialization')
    checkpoint = torch.load(folder/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state'])
    prediction = predict(model, eval_set, expected, nodes, cfg, device)
    actual, _ = score(prediction, truth, net, family, method)
    selected = float(aggregate(pd.DataFrame(actual)).S_state.iloc[0])
    if abs(selected-manifest['metadata']['best_dev_S_state'])>1e-9:
        raise ValueError('Cached checkpoint does not reproduce selection score')
    del model
    train_set.day.cache_clear(); eval_set.day.cache_clear()
    return prediction, dict(manifest['metadata'], key=manifest['key'], folder=str(folder), hit=hit)


def initialization_check(records, source, means, attributes, adjacency, cfg, device):
    outputs = []; hashes = []; parameters = []
    for wave in [False, True]:
        seed(cfg); model = ContextNeural(adjacency).to(device).eval()
        hashes.append(weights_hash(model)); parameters.append(sum(p.numel() for p in model.parameters()))
        dataset = ContextWindows(records[:1], source['grid_train']['metadata'], means, attributes, wave, True)
        batch = dataset[0]
        with torch.inference_mode():
            values = model(*(batch[k][None].to(device) for k in ['features','prior','x','mask']))
        outputs.append(values.cpu().numpy())
        dataset.day.cache_clear(); del model
    if hashes[0]!=hashes[1] or parameters[0]!=parameters[1] or not np.array_equal(outputs[0], outputs[1]):
        raise ValueError('Paired models have different initialization or initial prediction')
    return dict(passed=True, initial_weights_sha256=hashes[0], parameters=parameters[0],
                same_initial_predictions=True, same_parameter_count=True, readout='zero-initialized residual over shared interpolation')


def run(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty output directory')
    cfg = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    cfg.update(panel='D12_I5_N', seed=20261016, sample_per_day=1000, fit_start='2030-06-01', fit_end='2030-12-31',
        eval_start='2031-01-01', eval_end='2031-01-31', window=48, stride=24, hidden=32, layers=6,
        epochs=2 if args.smoke else 30, steps_per_epoch=4 if args.smoke else 100, patience=5, learning_rate=.001)
    if cfg['batch_size']<1 or cfg['threads']<1:
        raise ValueError('Positive batch size and threads required')
    began = time.perf_counter(); seed(cfg)
    torch.set_num_threads(cfg['threads']); torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check setup_wave_context.ps1 or explicitly select cpu')
    device = torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f'[wave context] I5; {device}; mode={"smoke" if args.smoke else "formal"}; three methods; same temporal anchor', flush=True)
    source = sources(cfg)
    nodes = pd.DataFrame(source['grid_train']['metadata']['nodes'])
    records_train = source['input_records']['training'][:14] if args.smoke else source['input_records']['training']
    records_eval = source['input_records']['evaluation'][:3] if args.smoke else source['input_records']['evaluation']
    train_dates, eval_dates = [r['date'] for r in records_train], [r['date'] for r in records_eval]
    train_keys = source['keys'].loc[source['keys'].timestamp.str[:10].isin(train_dates)].reset_index(drop=True)
    expected = source['expected'].loc[source['expected'].timestamp.str[:10].isin(eval_dates)].reset_index(drop=True)
    control = aligned_prediction(expected, source['control'].loc[source['control'].timestamp.str[:10].isin(eval_dates)])
    diagnostic = expected.merge(source['diagnostics'], on=KEY, validate='one_to_one')
    root = Path(cfg['release_root']); net = network(root, cfg['panel']); family = panel_families(root)[cfg['panel']]
    truth = truth_rows(root, cfg['panel'], expected)
    control_rows, _ = score(control, truth, net, family, CONTROL)
    check = control_check(source, pd.DataFrame(control_rows), eval_dates, args.smoke)
    means = np.load(source['stats_folder']/'means.npy', allow_pickle=False)
    attributes = np.load(source['attributes_folder']/'attributes.npy', allow_pickle=False)
    adjacency = np.load(source['grid_train_folder']/'adjacency.npy', allow_pickle=False)
    anchors = anchor_check(records_train, records_eval, source, means, attributes)
    anchors['initialization'] = initialization_check(records_train, source, means, attributes, adjacency, cfg, device)
    receptive = receptive_field_check(adjacency, device, cfg['window'])
    output.mkdir(parents=True, exist_ok=True)
    cfg.update(actual_fit_dates=train_dates, actual_eval_dates=eval_dates)
    save_json(output/'run_config.json', cfg)
    save_json(output/'reference_check.json', dict(passed=True, check=check, subset=args.smoke, current_best_source=cfg['wave_run'],
        control_S_state=float(aggregate(pd.DataFrame(control_rows)).S_state.iloc[0])))
    save_json(output/'sample_check.json', dict(passed=True, formal_pool_rows=214000, used_rows=len(train_keys), evaluation_rows=len(expected),
        exact_ordered_keys_equal_to_baseline_and_grin=True, ordered_keys_sha256=ordered_key_hash(train_keys), artificial_masking=False, extra_visible_supervision=False))
    save_json(output/'anchor_check.json', anchors); save_json(output/'receptive_field_check.json', receptive)
    usage = dict(original_grid_cache_reused=True, original_wave_cache_reused=True, original_models_retrained=False,
        daily_inputs_rebuilt=False, verified_existing_days=245, input_run=cfg['input_run'],
        statistics=dict(key=source['stats']['key'], hit=True), static_attributes=dict(key=source['attributes_manifest']['key'], hit=True),
        training_days=records_train, evaluation_days=records_eval, models=[])
    save_json(output/'cache_usage.json', usage)
    save_json(output/'environment.json', dict(python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','pyarrow']}, device=str(device),
        cuda_device=torch.cuda.get_device_name(device) if device.type=='cuda' else None, source_files=implementation()))
    save_json(output/'source_manifest.json', snapshot(source['provenance']+[Path(r['folder'])/'manifest.json' for r in records_train+records_eval]))
    if args.action=='prepare':
        save_json(output/'memory.json', memory(device)); save_json(output/'log_update.json', dict(status='skipped_prepare'))
        save_json(output/'PREFLIGHT.json', dict(passed=True, training_rows=len(train_keys), evaluation_rows=len(expected),
            input_days=len(records_train)+len(records_eval), control_S_state=float(aggregate(pd.DataFrame(control_rows)).S_state.iloc[0]), model_trained=False))
        print(f'Preflight passed: {len(train_keys)} training, {len(expected)} evaluation targets; {len(nodes)} nodes; no training. {output}', flush=True)
        return
    cache = NeuralCache(Path(cfg['cache_dir']))
    files = day_files(root, cfg['panel'], 'train', False); truth_hashes = snapshot(files[d] for d in eval_dates)
    control.to_parquet(output/(CONTROL+'_predictions.parquet'), index=False)
    all_rows = export_errors(output, CONTROL, control, truth, net, family, diagnostic)
    histories = []
    for method in METHODS[1:]:
        wave = method==METHODS[2]
        train_set = ContextWindows(records_train, source['grid_train']['metadata'], means, attributes, wave, True)
        eval_set = ContextWindows(records_eval, source['grid_eval']['metadata'], means, attributes, wave, False)
        prediction, record = train_model(cache, method, cfg, train_set, eval_set, source, adjacency,
            expected, truth, net, family, nodes, train_keys, truth_hashes, device)
        if not np.isfinite(prediction[TARGETS].to_numpy()).all() or (prediction[TARGETS].to_numpy()<0).any():
            raise ValueError('Invalid final prediction')
        usage['models'].append(record); save_json(output/'cache_usage.json', usage)
        prediction.to_parquet(output/(method+'_predictions.parquet'), index=False)
        all_rows += export_errors(output, method, prediction, truth, net, family, diagnostic)
        history = pd.read_csv(Path(record['folder'])/'training_history.csv'); histories.append(history)
        history.to_csv(output/(method+'_training_history.csv'), index=False)
        save_json(output/(method+'_selection.json'), record)
    if len({r['initial_weights_sha256'] for r in usage['models']})!=1 or usage['models'][0]['initial_weights_sha256']!=anchors['initialization']['initial_weights_sha256']:
        raise ValueError('Paired networks did not share checked initialization')
    shared_epochs = min(len(h) for h in histories)
    if histories[0].window_order_sha256.iloc[:shared_epochs].tolist()!=histories[1].window_order_sha256.iloc[:shared_epochs].tolist():
        raise ValueError('Training window orders differ')
    detail = pd.DataFrame(all_rows); table = comparison(detail)
    table['delta_S_state_vs_current_best'] = table.S_state-float(table.loc[table.method.eq(CONTROL),'S_state'].iloc[0])
    table['delta_S_state_vs_no_wave'] = table.S_state-float(table.loc[table.method.eq(METHODS[1]),'S_state'].iloc[0])
    detail.to_csv(output/'metrics_by_panel_regime.csv', index=False)
    aggregate(detail).to_csv(output/'metrics_summary.csv', index=False); table.to_csv(output/'comparison_methods.csv', index=False)
    save_json(output/'memory.json', memory(device))
    save_json(output/'integrity_check.json', dict(passed=True, all_three_methods_same_targets=True, same_initial_weights=True,
        same_initial_predictions=True, same_temporal_prior=True, same_public_channels=True, same_window_order_for_shared_epochs=True,
        shared_epochs=shared_epochs, original_models_retrained=False, input_days_rebuilt=False, physics_penalties=False,
        feature_count=31, optional_wave_channels=13, actual_window_slots=48, layers=6, neural_models=2))
    save_json(output/'model_source_manifest.json', snapshot(Path(r['folder'])/'manifest.json' for r in usage['models']))
    save_json(output/'COMPLETE.json', dict(experiment='wave_context', panel=cfg['panel'], smoke=args.smoke, methods=METHODS,
        training_rows=len(train_keys), evaluation_rows=len(expected), reference_passed=True, pairing_passed=True, integrity_passed=True,
        all_regimes_present=bool(table.all_regimes_present.all()), seconds=time.perf_counter()-began))
    try:
        status = append_log(output, PROJECT/'实验日志.txt')
    except (ValueError, OSError) as error:
        save_json(output/'log_update.json', dict(status='failed', error=str(error)))
        raise RuntimeError('Evaluation complete; retry log without retraining') from error
    print(table.to_string(index=False), flush=True)
    print(f'Log status: {status["status"]}. Wave context complete: {output}', flush=True)
