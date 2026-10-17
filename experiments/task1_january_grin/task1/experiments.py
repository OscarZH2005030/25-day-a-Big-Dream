"""Controlled, cache-backed Task 1 feature ablations."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .cache import FeatureCache
from .cache_builder import (IDENTIFIER_COLUMNS, RESIDUAL_COLUMNS,
                            evaluation_cache, training_cache)
from .data import KEY, TARGETS, network, save_json, sha256
from .features import feature_columns
from .metrics import aggregate, score, validate_predictions
from .pipeline import (load_booster, model_parameters, panel_families,
                       save_booster, truth_rows)

METHODS = {'baseline': 'B2_lightgbm', 'multihop': 'B2_multihop', 'congestion': 'B2_congestion'}


def fit_cached(folder, training_manifest, destination, cfg, variant,
               target_mode='residual', label_folder=None, metadata_name='model_config.json'):
    began = time.perf_counter()
    table = pd.read_parquet(folder / 'training.parquet')
    cols = feature_columns(training_manifest['metadata']['feature_columns'], variant)
    x = table[cols]
    if target_mode == 'residual':
        y = table[RESIDUAL_COLUMNS].to_numpy(dtype=float)
    elif target_mode == 'direct':
        from .target_labels import aligned_direct_targets
        if label_folder is None:
            raise ValueError('Direct fitting requires an exact training-truth sidecar')
        labels = pd.read_parquet(label_folder / 'direct_targets.parquet')
        y = aligned_direct_targets(table, labels)
    else:
        raise ValueError(f'Unknown target_mode: {target_mode}')
    # Prevent an old residual-only prediction entry point from treating direct
    # models as residual models. Target comparisons use explicit target metadata.
    if target_mode == 'direct' and metadata_name == 'model_config.json':
        metadata_name = 'target_model_config.json'
    sample_hash = sha256(folder / 'training.parquet')
    params = model_parameters(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    for ch, name in enumerate(['speed', 'flow_lane']):
        print(f'[{variant}/{target_mode}] train {name}: {len(x):,} rows, {len(cols)} features', flush=True)
        model = lgb.train(params, lgb.Dataset(x, label=y[:, ch]), num_boost_round=cfg['rounds'])
        save_booster(model, destination / f'{name}.txt')
        pd.DataFrame({'feature': model.feature_name(), 'gain': model.feature_importance('gain')}).sort_values(
            'gain', ascending=False).to_csv(destination / f'{name}_importance.csv', index=False)
    shutil.copyfile(folder / 'profile.pkl', destination / 'profile.pkl')
    metadata = dict(config=cfg, feature_variant=variant, features=cols,
                    fit_dates=training_manifest['metadata']['fit_dates'], training_rows=len(table),
                    parameters=params, training_cache_key=training_manifest['key'],
                    training_table_sha256=sample_hash, target_mode=target_mode,
                    direct_targets_sha256=sha256(label_folder / 'direct_targets.parquet') if label_folder else None,
                    fit_seconds=time.perf_counter() - began)
    save_json(destination / metadata_name, metadata)
    del table, x, y, model
    gc.collect()


def compose_target_prediction(initial, model_output, target_mode):
    if target_mode == 'residual':
        return initial + model_output
    if target_mode == 'direct':
        return model_output
    raise ValueError(f'Unknown target_mode: {target_mode}')


def predict_cached(evaluation_folder, manifest, model_folders, cfg, specifications=None):
    if specifications is None:
        specifications = {v: dict(feature_variant=v, target_mode='residual', method=METHODS[v])
                          for v in model_folders}
    if set(specifications) != set(model_folders):
        raise ValueError('Model specifications do not match model folders')
    models = {v: [load_booster(folder / f'{c}.txt') for c in ['speed', 'flow_lane']]
              for v, folder in model_folders.items()}
    method_names = [specifications[v]['method'] for v in models]
    if len(set(method_names)) != len(method_names):
        raise ValueError('Duplicate experiment method names')
    outputs = {m: [] for m in ['B0_history', 'B1_interpolation'] + method_names}
    key_parts, diagnostic_parts = [], []
    for i, filename in enumerate(manifest['metadata']['files'], 1):
        table = pd.read_parquet(evaluation_folder / filename)
        keys = table[IDENTIFIER_COLUMNS].rename(columns={c: c[2:] for c in IDENTIFIER_COLUMNS})
        initial = table[['__initial_speed', '__initial_flow_lane']].to_numpy(dtype=float)
        lanes = table['__lanes'].to_numpy(dtype=float)
        candidates = {'B0_history': table[['__history_speed', '__history_flow_lane']].to_numpy(dtype=float),
                      'B1_interpolation': initial}
        for model_id, pair in models.items():
            spec = specifications[model_id]
            cols = feature_columns(manifest['metadata']['feature_columns'], spec['feature_variant'])
            x = table[cols]
            predicted = np.column_stack([m.predict(x, num_threads=cfg['threads']) for m in pair])
            candidates[spec['method']] = compose_target_prediction(initial, predicted, spec['target_mode'])
        for method, values in candidates.items():
            values = np.maximum(values, 0).copy()
            values[:, 1] *= lanes
            result = keys.copy()
            result[TARGETS] = values
            outputs[method].append(result)
        gap = table.speed_minutes_before + table.speed_minutes_after
        diagnostics = keys.copy()
        diagnostics['gap'] = pd.cut(gap, [0, 15, 30, 60, np.inf], labels=[
            '<=15min', '15-30min', '30-60min', '>60min']).astype(str)
        diagnostics.loc[~np.isfinite(gap), 'gap'] = 'one_side_or_none'
        diagnostic_parts.append(diagnostics)
        key_parts.append(keys)
        if i % 5 == 0 or i == len(manifest['metadata']['files']):
            print(f'[cached prediction] {i}/{len(manifest["metadata"]["files"])} days', flush=True)
    expected = pd.concat(key_parts, ignore_index=True)
    results = {method: pd.concat(parts, ignore_index=True) for method, parts in outputs.items()}
    for result in results.values():
        validate_predictions(expected, result)
    return expected, results, pd.concat(diagnostic_parts, ignore_index=True)


def evaluate_cached(root, panel, family, evaluation_folder, manifest, model_folders, cfg, output,
                    specifications=None):
    expected, predictions, diagnostics = predict_cached(evaluation_folder, manifest, model_folders, cfg, specifications)
    truth = truth_rows(root, panel, expected)
    net, rows = network(root, panel), []
    output.mkdir(parents=True, exist_ok=True)
    for method, pred in predictions.items():
        pred.to_parquet(output / f'{method}_predictions.parquet', index=False)
        detail, cells = score(pred, truth, net, family, method)
        rows.extend(detail)
        cells = cells.merge(diagnostics, on=KEY, how='left', validate='one_to_one')
        for name, grouping in [('daily', ['date', 'mask_regime']),
                                ('congestion', ['congested']), ('gap', ['gap'])]:
            summary = cells.groupby(grouping, as_index=False).agg(n=('speed_sq_error', 'size'),
                       speed_mse=('speed_sq_error', 'mean'), flow_lane_mse=('flow_lane_sq_error', 'mean'))
            summary.to_csv(output / f'{method}_{name}_errors.csv', index=False)
    return rows


def compare_to_reference(reference, cfg, panel, current):
    old_cfg = json.loads((reference / 'run_config.json').read_text(encoding='utf-8'))
    checked = ['fit_start', 'fit_end', 'max_fit_days', 'sample_per_day', 'rounds', 'threads',
               'max_gap_minutes', 'seed', 'eval_start', 'eval_end', 'max_eval_days']
    differences = {k: [old_cfg.get(k), cfg.get(k)] for k in checked if old_cfg.get(k) != cfg.get(k)}
    if differences:
        raise ValueError(f'Reference experiment settings differ: {differences}')
    original = pd.read_csv(reference / 'metrics_by_panel_regime.csv')
    original = original.loc[(original.panel == panel) & (original.method == 'B2_lightgbm')]
    control = current.loc[(current.panel == panel) & (current.method == 'B2_lightgbm')]
    if set(original.regime) != set(control.regime) or original.empty:
        raise ValueError(f'Reference does not contain matching panel/regimes: {panel}')
    joined = original.merge(control, on=['panel', 'regime'], suffixes=('_reference', '_control'), validate='one_to_one')
    deltas = {}
    for col in ['n', 'rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']:
        deltas[col] = float(abs(joined[col + '_reference'] - joined[col + '_control']).max())
    passed = deltas['n'] == 0 and max(deltas[c] for c in deltas if c != 'n') <= 1e-7
    return dict(panel=panel, passed=passed, max_absolute_difference=deltas,
                reference=str(reference.resolve()), tolerance=1e-7)


def comparison_table(detail):
    overall = aggregate(detail)
    result = overall.copy()
    base_score = float(overall.loc[overall.method.eq('B2_lightgbm'), 'S_state'].iloc[0])
    # Same hierarchy as scoring; RMSE is descriptive macro aggregation, not pooled RMSE.
    rmse = detail.groupby(['method', 'family', 'panel'], as_index=False)[
        ['rmse_speed_kmh', 'rmse_flow_per_lane']].mean().groupby(['method', 'family'], as_index=False)[
        ['rmse_speed_kmh', 'rmse_flow_per_lane']].mean().groupby('method', as_index=False)[
        ['rmse_speed_kmh', 'rmse_flow_per_lane']].mean()
    result = result.merge(rmse, on='method')
    result['delta_S_state_vs_B2'] = result.S_state - base_score
    return result


def ablate(args):
    root, output = args.release_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Ablation output must be empty; use a new directory')
    if args.fit_end >= args.eval_start:
        raise ValueError('Fit dates must precede evaluation dates')
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'func'}
    families = panel_families(root)
    panels = list(families) if args.panels == ['all'] else args.panels
    if set(panels) - set(families):
        raise ValueError('Unknown panels')
    variants = ['baseline'] + [v for v in args.variants if v != 'baseline']
    variants = list(dict.fromkeys(variants))
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'run_config.json', cfg)
    save_json(output / 'environment.json', dict(python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow', 'lightgbm', 'scipy']},
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    cache = FeatureCache(args.cache_dir.resolve())
    rows, usage, reference_checks = [], [], []
    for panel in panels:
        train_folder, train_manifest, train_hit = training_cache(cache, root, panel, cfg)
        eval_folder, eval_manifest, eval_hit = evaluation_cache(cache, root, panel, cfg, train_folder, train_manifest)
        usage.append(dict(panel=panel, training_key=train_manifest['key'], training_hit=train_hit,
                          evaluation_key=eval_manifest['key'], evaluation_hit=eval_hit))
        save_json(output / 'cache_usage.json', usage)
        folders = {}
        for variant in variants:
            destination = output / 'models' / variant / panel
            fit_cached(train_folder, train_manifest, destination, cfg, variant)
            folders[variant] = destination
        rows.extend(evaluate_cached(root, panel, families[panel], eval_folder, eval_manifest,
                                     folders, cfg, output / 'evaluation' / panel))
        detail = pd.DataFrame(rows)
        detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
        aggregate(detail).to_csv(output / 'metrics_summary.csv', index=False)
        compare = comparison_table(detail)
        compare.to_csv(output / 'comparison_vs_B2.csv', index=False)
        if args.reference_run:
            check = compare_to_reference(args.reference_run, cfg, panel, detail)
            reference_checks.append(check)
            save_json(output / 'reference_check.json', reference_checks)
            if not check['passed']:
                raise ValueError('Recomputed B2 does not reproduce reference. Inspect reference_check.json before comparing variants.')
        print(compare.to_string(index=False), flush=True)
    save_json(output / 'COMPLETE.json', dict(panels=panels, variants=variants,
              note='One-at-a-time feature ablations; all variants share training rows and model parameters.'))
    print(f'Ablations complete: {output}', flush=True)
