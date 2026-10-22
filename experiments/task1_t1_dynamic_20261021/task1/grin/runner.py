from __future__ import annotations

import importlib.metadata
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..cache import FeatureCache
from ..data import KEY, TARGETS, network, save_json, sha256
from ..metrics import aggregate, score, validate_predictions
from ..features import temporal_context
from ..pipeline import panel_families, truth_rows
from .data import GraphWindows, grid_cache, key_positions, time_covariates, window_starts
from .log import append_grin_log
from .model import GRIN, imputation_loss


def validate_protocol(cfg):
    required = dict(fit_start='2030-06-01', fit_end='2030-12-31',
        eval_start='2031-01-01', eval_end='2031-01-31', seed=20261016)
    if cfg['panel'] not in ['D12_I5_N','D7_I405_N'] or any(cfg[k] != v for k, v in required.items()):
        raise ValueError('GRIN uses D12_I5_N or D7_I405_N, June-December fit / January dev, seed 20261016')
    window_starts(cfg['window'], cfg['stride'])
    if not 0 <= cfg['drop_observed'] < 1 or not 0 <= cfg['aux_weight']:
        raise ValueError('Invalid masking/loss configuration')
    smoke = cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None
    if not smoke and cfg['sample_per_day'] != 1000:
        raise ValueError('Formal run keeps baseline sampling at 1000 targets/day')


def seed_and_device(cfg):
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(cfg['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run setup_grin.ps1 -TorchBuild cu128, or explicitly choose -Device cpu')
    device = torch.device('cuda' if cfg['device'] == 'cuda' or
        (cfg['device'] == 'auto' and torch.cuda.is_available()) else 'cpu')
    print(f'[GRIN] device={device}; torch={torch.__version__}', flush=True)
    return device


def blend_windows(predictions, starts, steps=288):
    """Positive triangular weights give every timestep coverage, including edges."""
    width = predictions.shape[1]
    weights = np.minimum(np.arange(width) + 1, width - np.arange(width)).astype(np.float64)
    output = np.zeros((steps, *predictions.shape[2:]), np.float64)
    count = np.zeros(steps, np.float64)
    for start, pred in zip(starts, predictions, strict=True):
        output[start:start + width] += pred * weights[:, None, None]
        count[start:start + width] += weights
    if (count <= 0).any():
        raise ValueError('Sliding-window inference left uncovered timesteps')
    return output / count[:, None, None]


@torch.inference_mode()
def predict(model, folder, manifest, cfg, device):
    model.eval()
    metadata = manifest['metadata']; nodes = pd.DataFrame(metadata['nodes'])
    expected = pd.read_parquet(folder / 'expected_keys.parquet')
    mean = np.asarray(metadata['scaler']['mean'], np.float32)
    std = np.asarray(metadata['scaler']['std'], np.float32)
    starts = window_starts(cfg['window'], cfg['stride'])
    outputs = []
    for date in metadata['dates']:
        with np.load(folder / (date + '.npz'), allow_pickle=False) as arrays:
            x, mask = arrays['x'], arrays['mask']
        normalized = np.where(mask, (x - mean) / std, 0).astype(np.float32)
        u = time_covariates(date, len(nodes))
        parts = []
        for first in range(0, len(starts), cfg['batch_size']):
            local = starts[first:first + cfg['batch_size']]
            tensors = [torch.from_numpy(np.stack([a[s:s + cfg['window']] for s in local])).to(device)
                       for a in [normalized, mask, u]]
            predictions, _ = model(*tensors)
            parts.append(predictions.cpu().numpy())
        values = blend_windows(np.concatenate(parts), starts)
        values = np.maximum(values * std + mean, 0)
        targets = expected.loc[expected.timestamp.str.startswith(date)].copy()
        rr, cc = key_positions(targets, nodes)
        values = values[rr, cc].copy()
        values[:, 1] *= nodes.lanes.to_numpy()[cc]
        targets[TARGETS] = values
        _, _, before, after, _ = temporal_context(np.where(mask[:,:,0], x[:,:,0], np.nan))
        gap = (before + after)[rr, cc]
        targets['gap'] = pd.cut(gap, [0,15,30,60,np.inf], labels=['<=15min','15-30min','30-60min','>60min']).astype(str)
        targets.loc[~np.isfinite(gap),'gap'] = 'one_side_or_none'
        outputs.append(targets)
    result = pd.concat(outputs, ignore_index=True)
    validate_predictions(expected, result)
    return result


def comparison(detail):
    table = aggregate(detail)
    columns = ['rmse_speed_kmh', 'rmse_flow_per_lane']
    errors = detail.groupby(['method', 'family', 'panel'], as_index=False)[columns].mean()
    errors = errors.groupby(['method', 'family'], as_index=False)[columns].mean().groupby('method', as_index=False)[columns].mean()
    table = table.merge(errors, on='method')
    baseline = table.loc[table.method.eq('Fusion_gap60mean')]
    if not baseline.empty:
        table['delta_S_state_vs_frozen_baseline'] = table.S_state - float(baseline.S_state.iloc[0])
    return table.sort_values('S_state', ascending=False)


def baseline_detail(cfg, grin_detail):
    if cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None:
        return pd.DataFrame()
    source = Path(cfg['baseline_run'])
    completed = json.loads((source / 'COMPLETE.json').read_text(encoding='utf-8'))
    if completed.get('experiment') != 'cross_corridor_review':
        raise ValueError('Frozen baseline must be the verified cross-corridor review')
    old_cfg = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    for key in ['fit_start', 'fit_end', 'eval_start', 'eval_end']:
        if old_cfg[key] != cfg[key]:
            raise ValueError('Frozen baseline date protocol differs')
    old = pd.read_csv(source / 'metrics_by_panel_regime.csv')
    old = old.loc[old.panel.eq(cfg['panel'])].copy()
    if 'Fusion_gap60mean' not in set(old.method):
        raise ValueError('Frozen baseline is missing')
    current_counts = grin_detail.set_index('regime').n.sort_index()
    for _, group in old.groupby('method'):
        if not group.set_index('regime').n.sort_index().equals(current_counts):
            raise ValueError('GRIN and baseline target coverage differs by regime')
    return old


def save_checkpoint(path, model, cfg, training, evaluation, epoch, value):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), config=cfg, training_key=training['key'],
        evaluation_key=evaluation['key'], best_epoch=epoch, dev_S_state=value), temporary)
    temporary.replace(path)


def baseline_sample_table(project, cfg):
    explicit = cfg.get('baseline_sample_table')
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_relative_to(project.resolve()):
            raise ValueError('Baseline sample table must stay inside project')
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if cfg['panel'] == 'D12_I5_N':
        return project / 'cache/task1/8101f3e541df5c3f35598ee209fb10900a2b25bde9839fc722446e733af77f13/training.parquet'
    source = project / 'runs/corridor_dev_20261016_192125/cache_usage.json'
    records = json.loads(source.read_text(encoding='utf-8'))
    record = next(r for r in records if r['panel'] == cfg['panel'])
    return project / 'cache/task1' / record['training_key'] / 'training.parquet'


def run(args):
    project = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new, empty result directory')
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    checkpoint = None
    if args.action == 'evaluate':
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        cfg = dict(checkpoint['config'], action='evaluate', checkpoint=str(args.checkpoint.resolve()),
                   output=str(output), device=args.device, threads=args.threads, no_log=args.no_log)
    validate_protocol(cfg)
    device = seed_and_device(cfg)
    root = Path(cfg['release_root']); cache = FeatureCache(Path(cfg['cache_dir']))
    train_folder, training, train_hit = grid_cache(cache, root, cfg, 'training')
    eval_folder, evaluation, eval_hit = grid_cache(cache, root, cfg, 'evaluation', parent=training)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'run_config.json', cfg)
    save_json(output / 'cache_usage.json', dict(training_key=training['key'], training_hit=train_hit,
        evaluation_key=evaluation['key'], evaluation_hit=eval_hit,
        evaluation_cache_contains_truth=False, training_scaler=training['metadata']['scaler']))
    save_json(output / 'environment.json', dict(python=__import__('platform').python_version(),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'pyarrow']},
        device=str(device), cuda_device=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        source_sha256={str(p.relative_to(project)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        metric_sha256=sha256(project / 'task1/metrics.py'),
        architecture_reference='https://github.com/Graph-Machine-Learning-Group/grin'))
    if args.action == 'prepare':
        print(f'GRIN grids prepared: {output}', flush=True)
        return
    # Confirm exact sampled IDs against the existing baseline cache when available.
    frozen = baseline_sample_table(project, cfg)
    if not cfg.get('max_fit_days') and frozen.exists():
        original = pd.read_parquet(frozen, columns=['__' + c for c in KEY]).rename(columns={ '__'+c:c for c in KEY})
        sampled = pd.read_parquet(train_folder / 'training_sample_keys.parquet')
        if not original.reset_index(drop=True).equals(sampled[KEY].reset_index(drop=True)):
            raise ValueError('GRIN training sample IDs differ from frozen baseline')
        save_json(output / 'sample_check.json', dict(passed=True, same_ordered_training_targets=True, rows=len(sampled)))
    adjacency = np.load(train_folder / 'adjacency.npy', allow_pickle=False)
    model = GRIN(adjacency, hidden=cfg['hidden'], dropout=cfg['dropout']).to(device)
    expected = pd.read_parquet(eval_folder / 'expected_keys.parquet')
    truth, net = truth_rows(root, cfg['panel'], expected), network(root, cfg['panel'])
    family = panel_families(root)[cfg['panel']]
    # Baseline compatibility is checked before spending the full training budget.
    counts = expected.groupby('mask_regime').size().rename_axis('regime').reset_index(name='n')
    baseline = baseline_detail(cfg, counts)
    best_score, best_epoch = -float('inf'), 0
    if checkpoint is not None:
        if checkpoint['training_key'] != training['key'] or checkpoint['evaluation_key'] != evaluation['key']:
            raise ValueError('Checkpoint and current grid cache identities differ')
        model.load_state_dict(checkpoint['model'])
        best_epoch, best_score = checkpoint['best_epoch'], checkpoint['dev_S_state']
    else:
        dataset = GraphWindows(train_folder, training, cfg['window'], cfg['stride'])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-5)
        channel_scale = torch.tensor(training['metadata']['scaler']['std'], device=device) / torch.tensor([25.,600.],device=device)
        history = []
        for epoch in range(1, cfg['epochs'] + 1):
            began = time.perf_counter(); model.train(); losses = []
            generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
            loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                                generator=generator, num_workers=0)
            for step, batch in enumerate(loader, 1):
                x, mask, u, y, supervised = [b.to(device) for b in batch]
                removed = (torch.rand((*mask.shape[:-1], 1), device=device) < cfg['drop_observed']) & mask
                target = torch.where(removed, x, y)
                active = supervised | removed
                optimizer.zero_grad(set_to_none=True)
                prediction, auxiliary = model(x, mask & ~removed, u)
                loss = imputation_loss(prediction, auxiliary, target, active, channel_scale, cfg['aux_weight'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step(); losses.append(float(loss.detach()))
                if step % 20 == 0:
                    print(f'[GRIN] epoch {epoch}/{cfg["epochs"]}, step {step}/{min(len(loader),cfg["steps_per_epoch"])}', flush=True)
                if step >= cfg['steps_per_epoch']:
                    break
            pred = predict(model, eval_folder, evaluation, cfg, device)
            rows, _ = score(pred, truth, net, family, 'GRIN')
            value = float(aggregate(pd.DataFrame(rows)).S_state.iloc[0])
            history.append(dict(epoch=epoch, training_loss=float(np.mean(losses)),
                                dev_S_state=value, seconds=time.perf_counter() - began))
            pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
            if value > best_score:
                best_score, best_epoch = value, epoch
                save_checkpoint(output / 'best.pt', model, cfg, training, evaluation, epoch, value)
            print(f'[GRIN] epoch={epoch}, loss={history[-1]["training_loss"]:.6f}, dev={value:.6f}, best_epoch={best_epoch}, seconds={history[-1]["seconds"]:.1f}', flush=True)
            if epoch - best_epoch >= cfg['patience']:
                break
        checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
    pred = predict(model, eval_folder, evaluation, cfg, device)
    pred.to_parquet(output / 'GRIN_predictions.parquet', index=False)
    rows, cells = score(pred, truth, net, family, 'GRIN')
    detail = pd.DataFrame(rows)
    if not baseline.empty:
        detail = pd.concat([detail, baseline], ignore_index=True)
    detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
    aggregate(detail).to_csv(output / 'metrics_summary.csv', index=False)
    comparison(detail).to_csv(output / 'comparison_vs_baseline.csv', index=False)
    for name, grouping in [('daily', ['date','mask_regime']), ('congestion', ['congested']), ('gap', ['gap'])]:
        cells.groupby(grouping, as_index=False).agg(n=('speed_sq_error','size'),
            speed_mse=('speed_sq_error','mean'), flow_lane_mse=('flow_lane_sq_error','mean')).to_csv(output / f'GRIN_{name}_errors.csv',index=False)
    save_json(output / 'selection.json', dict(best_epoch=best_epoch, best_dev_S_state=best_score,
        checkpoint_selection='January S_state; no January gradient updates; February untouched',
        training_rows=training['metadata']['sampled_training_rows'], nodes=len(adjacency),
        parameters=sum(p.numel() for p in model.parameters()), window=cfg['window'], stride=cfg['stride']))
    save_json(output / 'COMPLETE.json', dict(experiment='grin_dev', panel=cfg['panel'],
        methods=detail.method.unique().tolist(), evaluation_rows=len(pred), best_epoch=best_epoch))
    try:
        status = append_grin_log(output, project / '实验日志.txt')
    except (OSError, ValueError) as error:
        save_json(output / 'log_update.json', dict(status='failed', error=str(error)))
        raise RuntimeError('GRIN evaluation completed; run log action to retry log append') from error
    print(comparison(detail).to_string(index=False), flush=True)
    print(f'Log status: {status["status"]}. GRIN complete: {output}', flush=True)
