"""Three fixed feature sets x residual/direct regression, sharing feature caches."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from .cache import FeatureCache
from .cache_builder import training_cache, evaluation_cache
from .data import save_json, sha256
from .experiment_log import append_target_log
from .experiments import (METHODS, compare_to_reference, comparison_table,
                          evaluate_cached, fit_cached)
from .features import feature_columns
from .metrics import aggregate
from .pipeline import panel_families
from .target_labels import direct_label_cache

VARIANTS = ('baseline', 'multihop', 'congestion')
MODES = ('residual', 'direct')
DIRECT_METHODS = {v: 'Direct_' + v for v in VARIANTS}


def residual_reference_checks(reference, cfg, panel, current):
    results = [dict(feature_variant='baseline', **compare_to_reference(reference, cfg, panel, current))]
    old = pd.read_csv(reference / 'metrics_by_panel_regime.csv')
    for variant in VARIANTS[1:]:
        method = METHODS[variant]
        a = old.loc[(old.panel == panel) & (old.method == method)]
        b = current.loc[(current.panel == panel) & (current.method == method)]
        if a.empty or set(a.regime) != set(b.regime):
            raise ValueError(f'Reference must include residual {method} for {panel}')
        joined = a.merge(b, on=['panel', 'regime'], suffixes=('_old', '_new'), validate='one_to_one')
        differences = {c: float(abs(joined[c + '_old'] - joined[c + '_new']).max())
                       for c in ['n', 'rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']}
        passed = differences['n'] == 0 and max(differences[c] for c in differences if c != 'n') <= 1e-7
        results.append(dict(feature_variant=variant, panel=panel, passed=passed,
            max_absolute_difference=differences, reference=str(reference.resolve()), tolerance=1e-7))
    return results


def target_comparison(detail, feature_counts):
    summary = comparison_table(detail).set_index('method')
    result = []
    for variant in VARIANTS:
        residual = summary.loc[METHODS[variant]]
        direct = summary.loc[DIRECT_METHODS[variant]]
        delta = float(direct.S_state - residual.S_state)
        result.append(dict(feature_variant=variant, n_features=feature_counts[variant],
            residual_S_state=float(residual.S_state), direct_S_state=float(direct.S_state),
            delta_direct_minus_residual=delta,
            residual_speed_rmse=float(residual.rmse_speed_kmh), direct_speed_rmse=float(direct.rmse_speed_kmh),
            residual_flow_rmse=float(residual.rmse_flow_per_lane), direct_flow_rmse=float(direct.rmse_flow_per_lane),
            winner='tie' if abs(delta) <= 1e-9 else ('direct' if delta > 0 else 'residual')))
    return pd.DataFrame(result)


def target_compare(args):
    root, output = args.release_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty --output directory')
    if args.fit_end >= args.eval_start:
        raise ValueError('Fit dates must precede evaluation dates')
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'func'}
    families = panel_families(root)
    panels = list(families) if args.panels == ['all'] else args.panels
    if set(panels) - set(families):
        raise ValueError('Unknown panels')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'run_config.json', cfg)
    save_json(output / 'environment.json', dict(python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow', 'lightgbm', 'scipy']},
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    cache = FeatureCache(args.cache_dir.resolve())
    rows, usages, checks, pair_records = [], [], [], []
    feature_counts = None
    for panel in panels:
        training_folder, training, train_hit = training_cache(cache, root, panel, cfg)
        eval_folder, evaluation, eval_hit = evaluation_cache(cache, root, panel, cfg, training_folder, training)
        label_folder, label_manifest, labels_hit = direct_label_cache(cache, root, panel, training_folder, training)
        counts = {v: len(feature_columns(training['metadata']['feature_columns'], v)) for v in VARIANTS}
        if feature_counts is not None and feature_counts != counts:
            raise ValueError('Feature counts differ between panels')
        feature_counts = counts
        usages.append(dict(panel=panel, training_key=training['key'], training_hit=train_hit,
            evaluation_key=evaluation['key'], evaluation_hit=eval_hit,
            direct_labels_key=label_manifest['key'], direct_labels_hit=labels_hit))
        save_json(output / 'cache_usage.json', usages)
        folders, specifications = {}, {}
        for variant in VARIANTS:
            for mode in MODES:
                model_id = variant + '__' + mode
                destination = output / 'models' / variant / mode / panel
                fit_cached(training_folder, training, destination, cfg, variant, target_mode=mode,
                    label_folder=label_folder if mode == 'direct' else None,
                    metadata_name='target_model_config.json')
                folders[model_id] = destination
                specifications[model_id] = dict(feature_variant=variant, target_mode=mode,
                    method=METHODS[variant] if mode == 'residual' else DIRECT_METHODS[variant])
                meta = json.loads((destination / 'target_model_config.json').read_text(encoding='utf-8'))
                pair_records.append(dict(panel=panel, feature_variant=variant, target_mode=mode,
                    training_rows=meta['training_rows'], features=meta['features'],
                    sample_sha256=meta['training_table_sha256'], parameters=meta['parameters']))
        # Enforce fairness before evaluating: same rows/parameters across all six,
        # and the exact same feature columns within each target-mode pair.
        local = [p for p in pair_records if p['panel'] == panel]
        if len({p['sample_sha256'] for p in local}) != 1 or len({
                json.dumps(p['parameters'], sort_keys=True) for p in local}) != 1:
            raise ValueError('Target comparison sample/parameter mismatch')
        for variant in VARIANTS:
            pair = [p for p in local if p['feature_variant'] == variant]
            if pair[0]['features'] != pair[1]['features']:
                raise ValueError('Residual/direct feature columns differ')
        save_json(output / 'pairing_check.json', dict(passed=True, records=pair_records))
        rows.extend(evaluate_cached(root, panel, families[panel], eval_folder, evaluation,
            folders, cfg, output / 'evaluation' / panel, specifications=specifications))
        detail = pd.DataFrame(rows)
        detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
        aggregate(detail).to_csv(output / 'metrics_summary.csv', index=False)
        comparison_table(detail).to_csv(output / 'comparison_vs_B2.csv', index=False)
        table = target_comparison(detail, feature_counts)
        table.to_csv(output / 'comparison_target_modes.csv', index=False)
        if args.reference_run:
            checks.extend(residual_reference_checks(args.reference_run, cfg, panel, detail))
            save_json(output / 'reference_check.json', checks)
            if not all(c['passed'] for c in checks):
                raise ValueError('Residual controls did not reproduce reference; inspect reference_check.json')
        print(table.to_string(index=False), flush=True)
    save_json(output / 'COMPLETE.json', dict(experiment='target_comparison', panels=panels,
        variants=list(VARIANTS), target_modes=list(MODES), pairing_passed=True,
        note='Fixed-budget comparison; no claims about the optimal hyperparameters for either target mode.'))
    try:
        status = append_target_log(output, Path(__file__).resolve().parents[1] / '实验日志.txt')
    except (OSError, ValueError) as error:
        save_json(output / 'log_update.json', dict(status='failed', error=str(error)))
        raise RuntimeError('Experiments completed but log append failed. Run log-target to retry without retraining.') from error
    print(f'Log status: {status["status"]}. Target comparison complete: {output}', flush=True)
