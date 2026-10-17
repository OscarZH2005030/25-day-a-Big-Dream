"""Validate fixed multihop target/fusion choices on an independent corridor."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import FeatureCache, snapshot
from .cache_builder import evaluation_cache, training_cache
from .data import KEY, day_files, network, save_json, sha256
from .experiment_log import append_corridor_log
from .experiments import fit_cached, predict_cached
from .features import feature_columns
from .fusion_experiments import (DIRECT, RESIDUAL, VALIDATION_RULES, aligned_prediction,
                                 cached_diagnostics, fuse_predictions, fusion_comparison)
from .metrics import aggregate, score
from .pipeline import model_parameters, panel_families, truth_rows
from .target_labels import direct_label_cache

METHODS = [RESIDUAL, DIRECT, *VALIDATION_RULES]


def artifact_hash(manifest, filename):
    return next(a['sha256'] for a in manifest['artifacts'] if a['name'] == filename)


def verify_pair(folder, training_manifest, label_manifest, cfg):
    records = []
    for mode in ['residual', 'direct']:
        record = json.loads((folder / mode / 'target_model_config.json').read_text(encoding='utf-8'))
        if record['target_mode'] != mode or record['feature_variant'] != 'multihop':
            raise ValueError('Incorrect cached model target or feature variant')
        if record['parameters'] != model_parameters(cfg) or record['config']['rounds'] != cfg['rounds']:
            raise ValueError('Cached model training parameters differ')
        if record['training_cache_key'] != training_manifest['key'] or record['training_table_sha256'] != artifact_hash(training_manifest, 'training.parquet'):
            raise ValueError('Cached model training samples differ')
        if record['training_rows'] != training_manifest['metadata']['training_rows'] or record['fit_dates'] != training_manifest['metadata']['fit_dates']:
            raise ValueError('Cached model training scope differs')
        expected_label_hash = artifact_hash(label_manifest, 'direct_targets.parquet') if mode == 'direct' else None
        if record['direct_targets_sha256'] != expected_label_hash:
            raise ValueError('Cached direct model label sidecar differs')
        records.append(record)
    expected_features = feature_columns(training_manifest['metadata']['feature_columns'], 'multihop')
    if records[0]['features'] != records[1]['features'] or records[0]['features'] != expected_features:
        raise ValueError('Residual/direct feature columns differ')
    return dict(passed=True, n_features=len(records[0]['features']), training_rows=records[0]['training_rows'],
                sample_sha256=records[0]['training_table_sha256'], parameters=records[0]['parameters'],
                rounds=cfg['rounds'], feature_variant='multihop', target_modes=['residual', 'direct'])


def paired_model_cache(cache, panel, training_folder, training, label_folder, labels, cfg):
    source = Path(__file__).parent
    identity = dict(kind='multihop_model_pair', panel=panel, schema='paired_multihop_v1',
        training_key=training['key'], direct_labels_key=labels['key'],
        parameters=model_parameters(cfg), rounds=cfg['rounds'],
        implementation=snapshot(source / name for name in ['corridor_validation.py', 'experiments.py', 'pipeline.py', 'features.py']),
        packages={p: importlib.metadata.version(p) for p in ['lightgbm', 'numpy', 'pandas']})

    def build(folder):
        for mode in ['residual', 'direct']:
            fit_cached(training_folder, training, folder / mode, cfg, 'multihop', target_mode=mode,
                       label_folder=label_folder if mode == 'direct' else None,
                       metadata_name='target_model_config.json')
        return verify_pair(folder, training, labels, cfg)

    result = cache.get_or_build(identity, build)
    verify_pair(result[0], training, labels, cfg)
    return result


def paired_prediction_cache(cache, panel, model_folder, models, evaluation_folder, evaluation, cfg):
    identity = dict(kind='multihop_predictions', panel=panel, schema='multihop_predictions_v1',
        model_pair_key=models['key'], evaluation_key=evaluation['key'], threads=cfg['threads'],
        implementation=snapshot([Path(__file__), Path(__file__).with_name('experiments.py')]),
        packages={p: importlib.metadata.version(p) for p in ['lightgbm', 'numpy', 'pandas', 'pyarrow']},
        target_units=['km/h', 'veh/h/link'], evaluation_truth_used=False)

    def build(folder):
        specifications = {mode: dict(feature_variant='multihop', target_mode=mode,
            method=RESIDUAL if mode == 'residual' else DIRECT) for mode in ['residual', 'direct']}
        expected, predictions, _ = predict_cached(evaluation_folder, evaluation,
            {m: model_folder / m for m in specifications}, cfg, specifications=specifications)
        for method in [RESIDUAL, DIRECT]:
            aligned_prediction(expected, predictions[method]).to_parquet(folder / f'{method}_predictions.parquet', index=False)
        diagnostics = cached_diagnostics(evaluation_folder, evaluation)
        aligned_prediction(diagnostics[KEY], predictions[RESIDUAL])
        diagnostics.to_parquet(folder / 'diagnostics.parquet', index=False)
        return dict(target_rows=len(expected), eval_dates=evaluation['metadata']['eval_dates'],
                    methods=[RESIDUAL, DIRECT], evaluation_truth_used=False)

    return cache.get_or_build(identity, build)


def check_protocol(cfg, families):
    panels = cfg['panels']
    if not panels or len(set(panels)) != len(panels) or set(panels) - set(families):
        raise ValueError('Panels must be known and unique')
    if any(families[p] == 'D12_I5' for p in panels):
        raise ValueError('This validation requires a different corridor family from D12_I5')
    fixed = dict(fit_start='2030-06-01', fit_end='2030-12-31', eval_start='2031-01-01',
                 eval_end='2031-01-31', seed=20261016, max_gap_minutes=60)
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError('Keep the fixed June-December / January validation protocol; February remains held out')
    smoke = cfg.get('max_fit_days') is not None or cfg.get('max_eval_days') is not None
    if not smoke and (cfg['rounds'] != 350 or cfg['sample_per_day'] != 1000):
        raise ValueError('Formal comparison requires the original 350 rounds and 1000 samples/day')


def corridor_validate(args):
    root, output = args.release_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty --output directory')
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'func'}
    families = panel_families(root)
    check_protocol(cfg, families)
    panels = args.panels
    # Detect incomplete local data before starting expensive feature construction.
    for panel in panels:
        for start, end, limit in [(cfg['fit_start'], cfg['fit_end'], cfg.get('max_fit_days')),
                                  (cfg['eval_start'], cfg['eval_end'], cfg.get('max_eval_days'))]:
            dates = pd.date_range(start, end).strftime('%Y-%m-%d').tolist()
            if limit:
                dates = dates[:limit]
            for masked in [True, False]:
                if set(dates) - set(day_files(root, panel, 'train', masked)):
                    raise ValueError(f'Missing required daily files for {panel}')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'run_config.json', dict(cfg, fusion_rules=VALIDATION_RULES,
        validation_type='Same procedure, independently trained on the new corridor; not zero-shot model transfer'))
    save_json(output / 'environment.json', dict(python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow', 'lightgbm']},
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    cache = FeatureCache(args.cache_dir.resolve())
    rows, usage, pairing, inputs, references = [], [], [], [], []
    for panel in panels:
        train_folder, train, train_hit = training_cache(cache, root, panel, cfg)
        eval_folder, evaluation, eval_hit = evaluation_cache(cache, root, panel, cfg, train_folder, train)
        label_folder, labels, label_hit = direct_label_cache(cache, root, panel, train_folder, train)
        model_folder, models, model_hit = paired_model_cache(cache, panel, train_folder, train, label_folder, labels, cfg)
        pair = verify_pair(model_folder, train, labels, cfg)
        pairing.append(dict(panel=panel, **pair))
        save_json(output / 'pairing_check.json', dict(passed=True, records=pairing))
        prediction_folder, prediction_manifest, prediction_hit = paired_prediction_cache(
            cache, panel, model_folder, models, eval_folder, evaluation, cfg)
        usage.append(dict(panel=panel, training_key=train['key'], training_hit=train_hit,
            evaluation_key=evaluation['key'], evaluation_hit=eval_hit,
            direct_labels_key=labels['key'], direct_labels_hit=label_hit,
            model_pair_key=models['key'], model_pair_hit=model_hit,
            prediction_key=prediction_manifest['key'], prediction_hit=prediction_hit))
        save_json(output / 'cache_usage.json', usage)
        save_json(output / 'artifact_locations.json', [dict(panel=u['panel'],
            model_folder=str(cache.root / u['model_pair_key']), prediction_folder=str(cache.root / u['prediction_key'])) for u in usage])
        diagnostics = pd.read_parquet(prediction_folder / 'diagnostics.parquet')
        expected = diagnostics[KEY]
        if len(expected) != evaluation['metadata']['target_rows'] or not expected.panel.eq(panel).all():
            raise ValueError('Cached prediction target scope differs')
        if sorted(expected.timestamp.str[:10].unique()) != evaluation['metadata']['eval_dates']:
            raise ValueError('Cached prediction evaluation dates differ')
        predictions = fuse_predictions(expected,
            pd.read_parquet(prediction_folder / f'{RESIDUAL}_predictions.parquet'),
            pd.read_parquet(prediction_folder / f'{DIRECT}_predictions.parquet'),
            diagnostics.gap_minutes, rules=VALIDATION_RULES)
        truth, net = truth_rows(root, panel, expected), network(root, panel)
        destination = output / 'evaluation' / panel
        destination.mkdir(parents=True, exist_ok=True)
        panel_rows = []
        for method, pred in predictions.items():
            if method in VALIDATION_RULES:
                pred.to_parquet(destination / f'{method}_predictions.parquet', index=False)
            detail, cells = score(pred, truth, net, families[panel], method)
            panel_rows.extend(detail)
            cells = cells.merge(diagnostics, on=KEY, how='left', validate='one_to_one')
            for name, grouping in [('daily', ['date', 'mask_regime']), ('congestion', ['congested']), ('gap', ['gap'])]:
                cells.groupby(grouping, as_index=False).agg(n=('speed_sq_error', 'size'),
                    speed_mse=('speed_sq_error', 'mean'), flow_lane_mse=('flow_lane_sq_error', 'mean')).to_csv(
                        destination / f'{method}_{name}_errors.csv', index=False)
        rows.extend(panel_rows)
        detail = pd.DataFrame(rows)
        detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
        aggregate(detail).to_csv(output / 'metrics_summary.csv', index=False)
        table = fusion_comparison(detail)
        mean_score = float(table.loc[table.method.eq('Fusion_mean50'), 'S_state'].iloc[0])
        table['delta_S_state_vs_mean50'] = table.S_state - mean_score
        table.to_csv(output / 'comparison_corridor.csv', index=False)
        per_panel = []
        for _, group in detail.groupby('panel', sort=True):
            compare = fusion_comparison(group)
            compare.insert(0, 'panel', group.panel.iloc[0])
            compare['delta_S_state_vs_mean50'] = compare.S_state - float(compare.loc[compare.method.eq('Fusion_mean50'), 'S_state'].iloc[0])
            per_panel.append(compare)
        pd.concat(per_panel, ignore_index=True).to_csv(output / 'comparison_by_panel.csv', index=False)
        inputs += [cache.root / u / 'manifest.json' for u in [train['key'], evaluation['key'], labels['key'], models['key'], prediction_manifest['key']]]
        full = day_files(root, panel, 'train', False)
        inputs += [full[d] for d in evaluation['metadata']['eval_dates']]
        references.append(dict(panel=panel, model_folder=str(model_folder), prediction_folder=str(prediction_folder), passed=True))
        print(table.to_string(index=False), flush=True)
    save_json(output / 'input_manifest.json', snapshot(inputs))
    save_json(output / 'integrity_check.json', dict(passed=True, records=references,
        note='Content-addressed artifact checks, paired sample/parameter checks, unique target alignment and finite prediction checks passed'))
    save_json(output / 'COMPLETE.json', dict(experiment='corridor_validation', panels=panels,
        methods=METHODS, fusion_rules=VALIDATION_RULES, pairing_passed=True, integrity_passed=True,
        note='Fixed rules from D12_I5_N; new corridor models trained locally with identical settings; no February evaluation'))
    try:
        status = append_corridor_log(output, Path(__file__).resolve().parents[1] / '实验日志.txt')
    except (OSError, ValueError) as error:
        save_json(output / 'log_update.json', dict(status='failed', error=str(error)))
        raise RuntimeError('Validation completed; log append failed. Run log-corridor to retry without training.') from error
    print(f'Log status: {status["status"]}. Corridor validation complete: {output}', flush=True)
