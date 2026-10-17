from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import pickle
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import (JOIN_KEY, KEY, TARGETS, day_files, fit_profile, network,
                   read_csv, read_day, save_json, sha256, template)
from .features import build_features, feature_columns
from .metrics import aggregate, score, validate_predictions


def save_booster(model, path: Path):
    # Windows native LightGBM file IO does not reliably support Unicode paths.
    path.write_text(model.model_to_string(), encoding='utf-8')


def load_booster(path: Path):
    return lgb.Booster(model_str=path.read_text(encoding='utf-8'))


def model_parameters(cfg):
    return dict(objective='regression', metric='rmse', learning_rate=0.05,
                num_leaves=31, max_depth=8, min_data_in_leaf=50, max_bin=127,
                feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                lambda_l2=1.0, seed=cfg['seed'], deterministic=True,
                force_col_wise=True, num_threads=cfg['threads'], verbosity=-1)


def selected_targets(root, panel, split, start=None, end=None, limit=None):
    f = template(root, panel, split)
    dates = f.timestamp.str[:10]
    if start:
        f = f.loc[dates.ge(start)]
    if end:
        f = f.loc[f.timestamp.str[:10].le(end)]
    if limit:
        days = sorted(f.timestamp.str[:10].unique())[:limit]
        f = f.loc[f.timestamp.str[:10].isin(days)]
    if f.empty:
        raise ValueError(f'No targets: {panel}/{split}/{start}/{end}')
    return f.reset_index(drop=True)


def select_rows(frame, target):
    """Preserve template order while mapping to the day's masked input rows."""
    locator = frame.reset_index(names='_row')[JOIN_KEY + ['_row']]
    joined = target[JOIN_KEY].merge(locator, on=JOIN_KEY, how='left', validate='one_to_one')
    if joined._row.isna().any():
        raise ValueError('Template target absent from observation grid')
    return joined._row.to_numpy(dtype=int)


def assert_hidden(masked, positions):
    channels = [c for c in TARGETS + ['occupancy', 'density_occ_linear_vehpkm'] if c in masked]
    if masked.iloc[positions][channels].notna().any().any():
        raise ValueError('A target has visible measured values: check release/mask')


def truth_rows(root, panel, targets):
    files = day_files(root, panel, 'train', False)
    result = []
    for date, group in targets.groupby(targets.timestamp.str[:10], sort=True):
        f = read_day(files[date])
        joined = group[KEY].merge(f[JOIN_KEY + TARGETS], on=JOIN_KEY,
                                  how='left', validate='one_to_one')
        if not np.isfinite(joined[TARGETS]).all().all():
            raise ValueError(f'Missing truth for template: {date}')
        result.append(joined)
    return pd.concat(result, ignore_index=True)


def fit_panel(root: Path, panel: str, folder: Path, cfg: dict):
    started = time.perf_counter()
    net = network(root, panel)
    topology = read_csv(root / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv')
    targets = selected_targets(root, panel, 'train', cfg['fit_start'], cfg['fit_end'],
                                cfg.get('max_fit_days'))
    selected_dates = sorted(targets.timestamp.str[:10].unique())
    # The same exact date set is used by the profile and the residual learner.
    profile = fit_profile(root, panel, selected_dates[0], selected_dates[-1])
    masked_files = day_files(root, panel, 'train')
    full_files = day_files(root, panel, 'train', False)
    xx, yy = [], []
    rng = np.random.default_rng(cfg['seed'])
    for n, (date, group) in enumerate(targets.groupby(targets.timestamp.str[:10], sort=True), 1):
        masked = read_day(masked_files[date])
        complete = read_day(full_files[date])
        positions = select_rows(masked, group)
        assert_hidden(masked, positions)
        if len(positions) > cfg['sample_per_day']:
            # Sample target rows before taking their labels, uniformly within a day.
            sample = np.sort(rng.choice(len(group), cfg['sample_per_day'], replace=False))
            group = group.iloc[sample]
            positions = positions[sample]
        profile.accumulate(complete, sign=-1)
        try:
            x, initial, _, lanes = build_features(masked, profile, net, topology,
                                                 cfg['max_gap_minutes'])
        finally:
            profile.accumulate(complete, sign=1)
        labels = group[JOIN_KEY].merge(complete[JOIN_KEY + TARGETS], on=JOIN_KEY,
                                       how='left', validate='one_to_one')[TARGETS].to_numpy(dtype=float, copy=True)
        labels[:, 1] /= lanes[positions]
        residual = labels - initial[positions]
        if not np.isfinite(residual).all():
            raise ValueError('Non-finite training residuals')
        xx.append(x.iloc[positions].copy())
        yy.append(residual)
        if n % 5 == 0 or n == len(selected_dates):
            print(f'[{panel}] features {n}/{len(selected_dates)} days', flush=True)
    x_train = pd.concat(xx, ignore_index=True)
    y_train = np.concatenate(yy)
    del xx, yy
    params = model_parameters(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    for ch, name in enumerate(['speed', 'flow_lane']):
        dataset = lgb.Dataset(x_train, label=y_train[:, ch], free_raw_data=True)
        model = lgb.train(params, dataset, num_boost_round=cfg['rounds'])
        save_booster(model, folder / f'{name}.txt')
        pd.DataFrame({'feature': model.feature_name(),
                      'gain': model.feature_importance(importance_type='gain')}).sort_values(
                          'gain', ascending=False).to_csv(folder / f'{name}_importance.csv', index=False)
        del dataset, model
    with (folder / 'profile.pkl').open('wb') as stream:
        pickle.dump(profile, stream)
    save_json(folder / 'model_config.json', dict(config=cfg, panel=panel,
              features=list(x_train.columns), training_rows=len(x_train), fit_dates=selected_dates,
              fit_seconds=time.perf_counter() - started, parameters=params,
              interpolation='within-day, max span; historical fallback; train profile leave-day-out'))
    del x_train, y_train
    gc.collect()


def predict_panel(root: Path, panel: str, folder: Path, split: str,
                  start=None, end=None, limit=None):
    metadata = json.loads((folder / 'model_config.json').read_text(encoding='utf-8'))
    cfg = metadata['config']
    targets = selected_targets(root, panel, split, start, end, limit)
    dates = set(targets.timestamp.str[:10])
    if dates.intersection(metadata['fit_dates']):
        raise ValueError('Refusing in-sample evaluation/prediction: fit and target dates overlap')
    # Only load profiles produced locally by this project, never untrusted pickles.
    with (folder / 'profile.pkl').open('rb') as stream:
        profile = pickle.load(stream)
    net = network(root, panel)
    topology = read_csv(root / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv')
    models = [load_booster(folder / f'{ch}.txt') for ch in ['speed', 'flow_lane']]
    masked_files = day_files(root, panel, split)
    outputs = {method: [] for method in ['B0_history', 'B1_interpolation', 'B2_lightgbm']}
    for n, (date, group) in enumerate(targets.groupby(targets.timestamp.str[:10], sort=True), 1):
        frame = read_day(masked_files[date])
        positions = select_rows(frame, group)
        assert_hidden(frame, positions)
        variant = metadata.get('feature_variant', 'baseline')
        x, initial, history, lanes = build_features(frame, profile, net, topology,
                        cfg['max_gap_minutes'], feature_set='base' if variant == 'baseline' else 'all')
        x = x.loc[:, feature_columns(x.columns, variant)].iloc[positions]
        if list(x.columns) != metadata['features']:
            raise ValueError('Feature schema differs from fitted model')
        corrected = initial[positions] + np.column_stack([
            model.predict(x, num_threads=cfg['threads']) for model in models])
        for method, values in [('B0_history', history[positions]),
                               ('B1_interpolation', initial[positions]),
                               ('B2_lightgbm', corrected)]:
            values = np.maximum(values, 0).copy()
            values[:, 1] *= lanes[positions]
            result = group[KEY].copy()
            result[TARGETS] = values
            outputs[method].append(result)
        if n % 10 == 0 or n == len(dates):
            print(f'[{panel}] {split} predict {n}/{len(dates)} days', flush=True)
    outputs = {k: pd.concat(v, ignore_index=True) for k, v in outputs.items()}
    for pred in outputs.values():
        validate_predictions(targets, pred)
    return targets, outputs, net


def evaluate_panel(root, panel, folder, output, start, end, limit, family):
    targets, outputs, net = predict_panel(root, panel, folder, 'train', start, end, limit)
    truth = truth_rows(root, panel, targets)
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for method, pred in outputs.items():
        pred.to_parquet(output / f'{method}_predictions.parquet', index=False)
        details, cells = score(pred, truth, net, family, method)
        rows.extend(details)
        daily = cells.groupby(['date', 'mask_regime'], as_index=False).agg(
            n=('speed_sq_error', 'size'), speed_mse=('speed_sq_error', 'mean'),
            flow_lane_mse=('flow_lane_sq_error', 'mean'))
        daily.to_csv(output / f'{method}_daily_errors.csv', index=False)
        congestion = cells.groupby('congested', as_index=False).agg(
            n=('speed_sq_error', 'size'), speed_mse=('speed_sq_error', 'mean'),
            flow_lane_mse=('flow_lane_sq_error', 'mean'))
        congestion.to_csv(output / f'{method}_congestion_errors.csv', index=False)
        # Bounded example exports support plots without retaining a second full truth layer.
        if method == 'B2_lightgbm':
            cells.head(10000).to_parquet(output / 'example_cells.parquet', index=False)
    return rows


def panel_families(root):
    contract = json.loads((root / 'config' / 'competition_contract.json').read_text(encoding='utf-8'))
    return {p: f['family_id'] for f in contract['scope']['corridor_families'] for p in f['directions']}


def run(args):
    root, output = args.release_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty; choose a new --output: {output}')
    if args.fit_end >= args.eval_start:
        raise ValueError('fit-end must precede eval-start')
    cfg = {k: v for k, v in vars(args).items() if k not in ['func']}
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    output.mkdir(parents=True, exist_ok=True)
    families = panel_families(root)
    panels = list(families) if args.panels == ['all'] else args.panels
    if set(panels) - set(families):
        raise ValueError('Unknown panel')
    save_json(output / 'run_config.json', cfg)
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    save_json(output / 'environment.json', dict(python=platform.python_version(),
              packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow', 'lightgbm', 'scipy']},
              source_sha256=source_hashes,
              contract_sha256=sha256(root / 'config' / 'competition_contract.json')))
    all_rows = []
    for panel in panels:
        print(f'[{panel}] fit begin', flush=True)
        fit_panel(root, panel, output / 'models' / panel, cfg)
        all_rows.extend(evaluate_panel(root, panel, output / 'models' / panel,
                                       output / 'evaluation' / panel,
                                       args.eval_start, args.eval_end, args.max_eval_days, families[panel]))
        # Save partial progress so an interrupted multi-panel run retains completed results.
        detail = pd.DataFrame(all_rows)
        detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
        overall = aggregate(detail)
        overall.to_csv(output / 'metrics_summary.csv', index=False)
        print(overall.to_string(index=False), flush=True)
    print(f'Completed. Results: {output}', flush=True)


def evaluate(args):
    root, models, out = args.release_root.resolve(), args.models.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Evaluation output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    rows, families = [], panel_families(root)
    panels = list(families) if args.panels == ['all'] else args.panels
    for panel in panels:
        rows.extend(evaluate_panel(root, panel, models / panel, out / panel,
                                   args.eval_start, args.eval_end, args.max_eval_days, families[panel]))
    detail = pd.DataFrame(rows)
    detail.to_csv(out / 'metrics_by_panel_regime.csv', index=False)
    overall = aggregate(detail)
    overall.to_csv(out / 'metrics_summary.csv', index=False)
    print(overall.to_string(index=False))


def predict(args):
    root = args.release_root.resolve()
    families = panel_families(root)
    panels = list(families) if args.panels == ['all'] else args.panels
    if args.output.exists():
        raise FileExistsError('Prediction CSV already exists; choose a new --output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for panel in panels:
        for split in args.splits:
            _, outputs, _ = predict_panel(root, panel, args.models / panel, split)
            outputs[args.method].to_csv(args.output, mode='w' if first else 'a',
                                         header=first, index=False)
            first = False
    print(f'Per-task state CSV saved: {args.output} (not yet a unified Kaggle upload)')
