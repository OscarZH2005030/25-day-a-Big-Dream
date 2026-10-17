"""CPU LightGBM residual baseline using the frozen Transformer target caches."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path(__file__).resolve().parents[1]/'full_dataset_training_v2'
SCALE = np.array([25., 600.])
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y, pred, regimes):
    rows = []
    for regime in ['R1', 'R2', 'R3']:
        keep = regimes == regime
        if not keep.any():
            raise ValueError('Missing regime: ' + regime)
        rmse = np.sqrt(np.mean((pred[keep] - y[keep]) ** 2, axis=0))
        sv, sq = max(0., 1-rmse[0]/25), max(0., 1-rmse[1]/600)
        rows.append(dict(method='lightgbm_residual', regime=regime, n=int(keep.sum()),
                         rmse_speed_kmh=float(rmse[0]), rmse_flow_per_lane=float(rmse[1]),
                         S_speed=sv, S_flow=sq, S_state=.54*sv+.46*sq))
    rows.append(dict(method='lightgbm_residual', regime='macro_mean', n=len(y),
                     S_state=float(np.mean([r['S_state'] for r in rows]))))
    return rows


def day_rows(cache, date):
    folder = cache/date
    ii = np.load(folder/'ii.npy', allow_pickle=False)
    jj = np.load(folder/'jj.npy', allow_pickle=False)
    x = np.load(folder/'features.npy', mmap_mode='r')[jj, ii].astype('float32')
    base = np.load(folder/'base.npy', mmap_mode='r')[jj, ii].astype('float64') * SCALE
    y = np.load(folder/'label.npy', mmap_mode='r')[jj, ii].astype('float64') * SCALE
    if np.isinf(x).any() or not np.isfinite(base).all() or not np.isfinite(y).all():
        raise ValueError('Invalid cached target values: ' + date)
    return x, base, y


def materialize(cache, dates, scratch, name):
    n = sum(len(np.load(cache/d/'ii.npy', mmap_mode='r')) for d in dates)
    x = np.lib.format.open_memmap(scratch/(name+'_x.npy'), mode='w+', dtype='float32', shape=(n, 51))
    residual = np.lib.format.open_memmap(scratch/(name+'_residual.npy'), mode='w+', dtype='float64', shape=(n, 2))
    cursor = 0
    for k, date in enumerate(dates):
        features, base, y = day_rows(cache, date)
        end = cursor + len(features)
        x[cursor:end] = features
        residual[cursor:end] = y-base
        cursor = end
        if k % 20 == 0 or k+1 == len(dates):
            print(f'{name}: {k+1}/{len(dates)} days; {cursor:,}/{n:,} targets', flush=True)
    x.flush(); residual.flush()
    return x, residual


def verify_panel(source, run, panel, plan):
    folder = run/panel
    meta = json.loads((folder/'cache/ready.json').read_text())
    signature = meta['signature']
    expected = dict(panel=panel, train_start=plan['training_dates'][0],
                    internal_val_start=plan['internal_validation_dates'][0],
                    val_start=plan['holdout_dates'][0], val_end='2031-03-01')
    for key, value in expected.items():
        if signature[key] != value:
            raise ValueError(f'Cache split mismatch: {panel} {key}')
    hashes = [digest(source/'full_train_worker.py'), digest(source/'task1_baseline.py')]
    if signature['code_sha256'] != hashes:
        raise ValueError('Cache/source hash mismatch: ' + panel)
    for item in signature['data_files']:
        stat = Path(item['path']).stat()
        if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError('Original data changed: ' + item['path'])
    dates = plan['training_dates'] + plan['internal_validation_dates'] + plan['holdout_dates']
    if meta['dates'] != dates or len(meta['names']) != 51:
        raise ValueError('Cache dates/features mismatch: ' + panel)
    history = json.loads((folder/'experiment.json').read_text())
    # Completed Transformer runs replace the initial history manifest with the
    # final experiment report. The frozen cache and source signature establish
    # the profile construction; verify the final report's training boundary too.
    if 'historical_start' in history:
        if history['historical_start'] != plan['history_dates'][0] or history['historical_end'] != plan['history_dates'][-1]:
            raise ValueError('Historical split mismatch: ' + panel)
    elif history['train_start'] != plan['training_dates'][0]:
        raise ValueError('Reference training boundary mismatch: ' + panel)
    if not (folder/'COMPLETE.json').exists():
        raise ValueError('Transformer reference not complete: ' + panel)
    return meta


def run_panel(args, plan, panel, meta, params):
    started = time.time()
    reference = args.reference_run/panel
    cache = reference/'cache'
    out = args.output/panel
    out.mkdir(exist_ok=True)
    scratch = out/'scratch'
    scratch.mkdir(exist_ok=True)
    x, y = materialize(cache, plan['training_dates'], scratch, 'train')
    vx, vy = materialize(cache, plan['internal_validation_dates'], scratch, 'internal')
    models, choices, importance = [], [], []
    for c, channel in enumerate(['speed_kmh', 'flow_per_lane']):
        print(f'{panel}: fit {channel}; {len(x):,} training targets, CPU only', flush=True)
        train = lgb.Dataset(x, label=np.asarray(y[:, c]).copy(), feature_name=meta['names'], free_raw_data=True)
        valid = lgb.Dataset(vx, label=np.asarray(vy[:, c]).copy(), reference=train, free_raw_data=True)
        evaluation = {}
        model = lgb.train(params, train, num_boost_round=args.max_trees,
                          valid_sets=[valid], valid_names=['internal'],
                          callbacks=[lgb.early_stopping(args.patience, first_metric_only=True),
                                     lgb.record_evaluation(evaluation), lgb.log_evaluation(50)])
        save_json(out/(channel+'_validation_trace.json'), evaluation)
        temp = out/(channel+'.tmp')
        model.save_model(str(temp), num_iteration=model.best_iteration)
        temp.replace(out/(channel+'.txt'))
        choices.append(dict(channel=channel, best_iteration=model.best_iteration,
                            internal_residual_rmse=float(model.best_score['internal']['rmse'])))
        importance.extend(dict(target=channel, feature=name, gain=float(gain))
                          for name, gain in zip(meta['names'], model.feature_importance('gain', iteration=model.best_iteration)))
        # Reload a predictor without retaining Dataset objects for both channels.
        models.append(lgb.Booster(model_file=str(out/(channel+'.txt'))))
        del model, train, valid
        gc.collect()
    del x, y, vx, vy
    gc.collect()
    # Holdout labels are accessed for scoring only after model selection finishes.
    predictions, truths, regimes, frames = [], [], [], []
    for date in plan['holdout_dates']:
        print(f'{panel}: holdout {date}', flush=True)
        features, base, truth = day_rows(cache, date)
        pred = np.maximum(base + np.column_stack([m.predict(features, num_threads=args.threads) for m in models]), 0)
        keys = pd.read_parquet(cache/date/'keys.parquet')
        if len(keys) != len(pred):
            raise ValueError('Target key length mismatch')
        frames.append(keys); predictions.append(pred); truths.append(truth)
        regimes.append(keys.mask_regime.to_numpy())
    pred = np.concatenate(predictions); truth = np.concatenate(truths)
    keys = pd.concat(frames, ignore_index=True)
    if keys.duplicated(KEYS).any() or not np.isfinite(pred).all():
        raise ValueError('Duplicate target keys/nonfinite predictions')
    reference_keys = pd.read_parquet(reference/'holdout_predictions.parquet', columns=KEYS)
    if not keys[KEYS].astype(str).equals(reference_keys[KEYS].astype(str)):
        raise ValueError('Holdout target keys/order differ from Transformer')
    del reference_keys
    rows = score(truth, pred, np.concatenate(regimes))
    topology = pd.read_csv(Path(plan['release_root'])/'corridors'/panel/'network/fd_parameters.csv').set_index('link_id')
    lanes = keys.link_id.map(topology.lanes).to_numpy()
    if not np.isfinite(lanes).all():
        raise ValueError('Missing lane counts')
    keys['speed_kmh'] = pred[:, 0]; keys['flow_vph'] = pred[:, 1]*lanes
    keys.to_parquet(out/'holdout_predictions.parquet', index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(importance).to_csv(out/'feature_importance.csv', index=False)
    save_json(out/'experiment.json', dict(panel=panel, feature_names=meta['names'],
              train_days=len(plan['training_dates']), internal_days=len(plan['internal_validation_dates']),
              holdout_days=len(plan['holdout_dates']), holdout_targets=len(keys), choices=choices,
              elapsed_seconds=time.time()-started, parameters=params,
              note='All cached template training targets; 51 raw features with native NaNs. Per-target internal RMSE early stopping. No holdout tuning or physics.'))
    for name in ['train_x.npy', 'train_residual.npy', 'internal_x.npy', 'internal_residual.npy']:
        (scratch/name).unlink()
    save_json(out/'COMPLETE.json', dict(panel=panel, S_state=rows[-1]['S_state']))
    print(f'FINISHED {panel}: {rows[-1]["S_state"]:.10f}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'runs/all_corridors_time80_20')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-trees', type=int, default=2000)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    if min(args.threads, args.max_trees, args.patience) < 1:
        p.error('threads, max-trees and patience must be positive')
    args.reference_run = args.reference_run or args.source/'runs/all_corridors_time80_20'
    plan = json.loads((args.reference_run/'plan.json').read_text())
    sys.path.insert(0, str(args.source))
    from training_io import RunLock
    params = dict(objective='regression', metric='rmse', learning_rate=.05, num_leaves=31,
                  min_data_in_leaf=80, lambda_l2=1., feature_fraction=.9, num_threads=args.threads,
                  seed=2026, deterministic=True, force_col_wise=True, verbosity=-1, device_type='cpu')
    print('Checking all ten frozen caches and source data metadata...', flush=True)
    metas = {panel: verify_panel(args.source, args.reference_run, panel, plan) for panel in plan['panels']}
    if args.preflight_only:
        # Real cached samples, no LightGBM training or result-directory writes.
        for panel, meta in metas.items():
            x, base, y = day_rows(args.reference_run/panel/'cache', plan['training_dates'][0])
            assert x.shape[1] == 51 and len(x) > 0
            print(f'PASS {panel}: {len(x):,} targets, 51 features', flush=True)
        print('PREFLIGHT PASSED. Formal training was not started.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with RunLock(args.output/'queue.lock'):
        config = dict(reference_run=str(args.reference_run.resolve()), plan_sha256=digest(args.reference_run/'plan.json'),
                      script_sha256=digest(Path(__file__)), parameters=params, max_trees=args.max_trees,
                      patience=args.patience, sampling='all template targets on supervised training dates',
                      feature_names=metas[plan['panels'][0]]['names'],
                      history_dates=plan['history_dates'], training_dates=plan['training_dates'],
                      internal_validation_dates=plan['internal_validation_dates'], holdout_dates=plan['holdout_dates'])
        manifest = args.output/'plan.json'
        if manifest.exists():
            if not args.resume or json.loads(manifest.read_text()) != config:
                raise ValueError('Existing run/config differs. Use --resume with unchanged code and options, or a new output.')
        else:
            save_json(manifest, config)
            (args.output/'training_source.py').write_bytes(Path(__file__).read_bytes())
            with (args.output/'environment_pip_freeze.txt').open('w') as f:
                subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
        completed = []
        try:
            for panel in plan['panels']:
                if (args.output/panel/'COMPLETE.json').exists():
                    completed.append(panel); print('SKIP ' + panel, flush=True); continue
                save_json(args.output/'status.json', dict(state='running', current=panel, completed=completed))
                with RunLock(args.output/(panel+'.lock')):
                    run_panel(args, plan, panel, metas[panel], params)
                completed.append(panel)
            all_rows = []
            for panel in plan['panels']:
                frame = pd.read_csv(args.output/panel/'metrics.csv'); frame.insert(0, 'panel', panel); all_rows.append(frame)
            metrics = pd.concat(all_rows, ignore_index=True)
            metrics.to_csv(args.output/'all_metrics.csv', index=False)
            baseline = metrics[metrics.regime == 'macro_mean'][['panel', 'S_state']].rename(columns={'S_state':'LightGBM'})
            ref = pd.read_csv(args.reference_run/'all_metrics.csv')
            compare = baseline.merge(ref[ref.regime == 'macro_mean'][['panel','S_state']].rename(columns={'S_state':'Transformer'}), on='panel', validate='one_to_one')
            compare['Transformer_minus_LightGBM'] = compare.Transformer-compare.LightGBM
            compare.to_csv(args.output/'comparison.csv', index=False)
            save_json(args.output/'summary.json', dict(corridors=len(completed),
                mean_corridor_S_state=float(baseline.LightGBM.mean()),
                note='Equal-corridor local Task 1 mean, not official total score; identical date split, different optimization/sampling budgets.'))
            save_json(args.output/'status.json', dict(state='complete', current=None, completed=completed))
            print(compare.to_string(index=False), flush=True)
        except BaseException as error:
            save_json(args.output/'status.json', dict(state='failed', current=panel, completed=completed, error=str(error)))
            raise


if __name__ == '__main__':
    main()
