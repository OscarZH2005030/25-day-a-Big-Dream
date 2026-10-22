"""Fixed multihop fusion rules using saved predictions; never fits a model."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import FeatureCache, snapshot
from .cache_builder import IDENTIFIER_COLUMNS
from .data import KEY, TARGETS, day_files, network, save_json, sha256
from .experiment_log import append_fusion_log
from .metrics import aggregate, score, validate_predictions
from .pipeline import panel_families, truth_rows

RESIDUAL = 'B2_multihop'
DIRECT = 'Direct_multihop'
RULES = {
    'Fusion_channels': 'speed=residual; flow=direct',
    'Fusion_mean50': '0.5 * residual + 0.5 * direct for both channels',
    'Fusion_gap30': 'speed bracketing span <=30min: residual; >30min or missing side: direct, both channels',
}
VALIDATION_RULES = {
    'Fusion_mean50': RULES['Fusion_mean50'],
    'Fusion_gap60mean': 'speed bracketing span >60min or missing side: direct; otherwise 50/50 mean, both channels',
}


def aligned_prediction(expected, predicted):
    if expected.empty or expected[KEY].isna().any().any() or expected.duplicated(KEY).any():
        raise ValueError('Expected targets must be nonempty, unique and non-null')
    if predicted[KEY].isna().any().any():
        raise ValueError('Null prediction keys')
    validate_predictions(expected, predicted)
    return expected[KEY].merge(predicted[KEY + TARGETS], on=KEY, how='left',
                               sort=False, validate='one_to_one')


def fuse_predictions(expected, residual, direct, gap_minutes, rules=None):
    """Inputs/outputs use km/h and TOTAL veh/h, exactly as saved predictions do.

    Gating uses only the observable speed bracketing span; the same gate applies
    to both outputs. Missing either side selects direct. There is no truth input.
    """
    residual = aligned_prediction(expected, residual)
    direct = aligned_prediction(expected, direct)
    gap = np.asarray(gap_minutes, dtype=float)
    if gap.shape != (len(expected),) or (gap[np.isfinite(gap)] <= 0).any():
        raise ValueError('Invalid observable gap vector')
    r, d = residual[TARGETS].to_numpy(float), direct[TARGETS].to_numpy(float)
    mean = 0.5 * r + 0.5 * d
    available = {
        'Fusion_channels': np.column_stack([r[:, 0], d[:, 1]]),
        'Fusion_mean50': mean,
        'Fusion_gap30': np.where((np.isfinite(gap) & (gap <= 30))[:, None], r, d),
        'Fusion_gap60mean': np.where((np.isfinite(gap) & (gap <= 60))[:, None], mean, d),
    }
    selected = list(RULES) if rules is None else list(rules)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(available):
        raise ValueError('Unknown, duplicate or empty fusion rules')
    outputs = {RESIDUAL: residual, DIRECT: direct}
    for method in selected:
        pred = expected[KEY].copy()
        pred[TARGETS] = available[method]
        validate_predictions(expected, pred)
        outputs[method] = pred
    return outputs


def cached_diagnostics(folder, manifest):
    parts = []
    for filename in manifest['metadata']['files']:
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Cache file escapes its directory')
        table = pd.read_parquet(path, columns=IDENTIFIER_COLUMNS +
                                ['speed_minutes_before', 'speed_minutes_after'])
        keys = table[IDENTIFIER_COLUMNS].rename(columns={c: c[2:] for c in IDENTIFIER_COLUMNS})
        gap = (table.speed_minutes_before + table.speed_minutes_after).to_numpy(float)
        if (gap[np.isfinite(gap)] <= 0).any():
            raise ValueError('Invalid cached bracketing span')
        keys['gap_minutes'] = gap
        keys['gap'] = pd.cut(gap, [0, 15, 30, 60, np.inf], labels=[
            '<=15min', '15-30min', '30-60min', '>60min']).astype(str)
        keys.loc[~np.isfinite(gap), 'gap'] = 'one_side_or_none'
        parts.append(keys)
    diagnostics = pd.concat(parts, ignore_index=True)
    if diagnostics[KEY].isna().any().any() or diagnostics.duplicated(KEY).any():
        raise ValueError('Invalid diagnostic target keys')
    if len(diagnostics) != manifest['metadata']['target_rows']:
        raise ValueError('Cache target count mismatch')
    dates = sorted(diagnostics.timestamp.str[:10].unique())
    if dates != manifest['metadata']['eval_dates']:
        raise ValueError('Cache evaluation date mismatch')
    return diagnostics


def control_reference_checks(reference, detail, panel):
    checks = []
    for method in [RESIDUAL, DIRECT]:
        a = reference.loc[(reference.panel == panel) & (reference.method == method)]
        b = detail.loc[(detail.panel == panel) & (detail.method == method)]
        if a.empty or a.regime.duplicated().any() or set(a.regime) != set(b.regime):
            raise ValueError(f'Missing or invalid source control: {panel}/{method}')
        paired = a.merge(b, on=['panel', 'regime'], validate='one_to_one', suffixes=('_old', '_new'))
        differences = {c: float(abs(paired[c + '_old'] - paired[c + '_new']).max())
                       for c in ['n', 'rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']}
        passed = all(np.isfinite(list(differences.values()))) and differences['n'] == 0 and max(
            differences[c] for c in differences if c != 'n') <= 1e-7
        checks.append(dict(panel=panel, method=method, passed=passed,
                           max_absolute_difference=differences, tolerance=1e-7))
    return checks


def fusion_comparison(detail):
    rmse_cols = ['rmse_speed_kmh', 'rmse_flow_per_lane']
    rmse = detail.groupby(['method', 'family', 'panel'], as_index=False)[rmse_cols].mean()
    rmse = rmse.groupby(['method', 'family'], as_index=False)[rmse_cols].mean()
    rmse = rmse.groupby('method', as_index=False)[rmse_cols].mean()
    table = aggregate(detail).merge(rmse, on='method')
    for control, suffix in [(RESIDUAL, 'residual'), (DIRECT, 'direct')]:
        base = table.loc[table.method.eq(control)].iloc[0]
        table['delta_S_state_vs_' + suffix] = table.S_state - base.S_state
        for column in rmse_cols:
            table['delta_' + column + '_vs_' + suffix] = table[column] - base[column]
    table['rule'] = table.method.map({RESIDUAL: 'residual control', DIRECT: 'direct control', **RULES, **VALIDATION_RULES})
    return table.sort_values('S_state', ascending=False).reset_index(drop=True)


def fusion_compare(args):
    source, output = args.source_run.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new, empty --output directory')
    completion = json.loads((source / 'COMPLETE.json').read_text(encoding='utf-8'))
    cfg = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    pairing = json.loads((source / 'pairing_check.json').read_text(encoding='utf-8'))
    if completion.get('experiment') != 'target_comparison' or not completion.get('pairing_passed') or not pairing.get('passed'):
        raise ValueError('Source must be a completed paired target comparison')
    if cfg.get('reference_run'):
        checks = json.loads((source / 'reference_check.json').read_text(encoding='utf-8'))
        if not checks or not all(c['passed'] for c in checks):
            raise ValueError('Source residual reference checks failed')
    root = Path(cfg['release_root']).resolve()
    if cfg['fit_end'] >= cfg['eval_start']:
        raise ValueError('Source fit/evaluation dates overlap')
    panels = completion['panels'] if args.panels == ['all'] else args.panels
    if not panels or len(set(panels)) != len(panels) or set(panels) - set(completion['panels']):
        raise ValueError('Requested panels must exist uniquely in source run')
    cache_dir = (args.cache_dir or Path(cfg['cache_dir'])).resolve()
    project = Path(__file__).resolve().parents[1]
    if not cache_dir.is_relative_to(project):
        raise ValueError('Cache directory must stay inside the project')
    usage = json.loads((source / 'cache_usage.json').read_text(encoding='utf-8'))
    reference = pd.read_csv(source / 'metrics_by_panel_regime.csv')
    run_cfg = dict(cfg, command='fusion-compare', source_run=str(source), output=str(output),
                   panels=panels, cache_dir=str(cache_dir), smoke=args.smoke, no_log=args.no_log,
                   fusion_rules=RULES, selection_note='Exploratory January development comparison; fixed rules, no threshold sweep.')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'run_config.json', run_cfg)
    save_json(output / 'environment.json', dict(python=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'pyarrow']},
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    input_paths = [source / name for name in ['COMPLETE.json', 'run_config.json', 'pairing_check.json',
                                             'metrics_by_panel_regime.csv', 'cache_usage.json']]
    rows, checks, reused = [], [], []
    families = panel_families(root)
    for panel in panels:
        matches = [u for u in usage if u['panel'] == panel]
        if len(matches) != 1:
            raise ValueError(f'Missing or ambiguous evaluation cache for {panel}')
        folder = cache_dir / matches[0]['evaluation_key']
        if not folder.resolve().is_relative_to(cache_dir):
            raise ValueError('Invalid evaluation cache key')
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        FeatureCache(cache_dir)._validate(folder, manifest['identity'])
        identity = manifest['identity']
        if manifest['key'] != matches[0]['evaluation_key'] or identity['panel'] != panel or identity['kind'] != 'evaluation' or Path(identity['release_root']).resolve() != root:
            raise ValueError('Source evaluation cache identity mismatch')
        if not all(cfg['eval_start'] <= d <= cfg['eval_end'] for d in manifest['metadata']['eval_dates']):
            raise ValueError('Cache dates outside source evaluation window')
        # Check the masked/network/template inputs behind observable diagnostics.
        for item in identity['inputs']:
            path = Path(item['path'])
            if not path.is_file() or sha256(path) != item['sha256']:
                raise ValueError(f'Source cache input changed: {path}')
        diagnostics = cached_diagnostics(folder, manifest)
        expected = diagnostics[KEY]
        paths = {m: source / 'evaluation' / panel / f'{m}_predictions.parquet' for m in [RESIDUAL, DIRECT]}
        predictions = fuse_predictions(expected, pd.read_parquet(paths[RESIDUAL]),
                                       pd.read_parquet(paths[DIRECT]), diagnostics.gap_minutes)
        truth = truth_rows(root, panel, expected)
        net = network(root, panel)
        destination = output / 'evaluation' / panel
        destination.mkdir(parents=True, exist_ok=True)
        panel_rows = []
        for method, pred in predictions.items():
            if method in RULES:
                pred.to_parquet(destination / f'{method}_predictions.parquet', index=False)
            detail, cells = score(pred, truth, net, families[panel], method)
            panel_rows.extend(detail)
            cells = cells.merge(diagnostics, on=KEY, how='left', validate='one_to_one')
            for name, grouping in [('daily', ['date', 'mask_regime']),
                                    ('congestion', ['congested']), ('gap', ['gap'])]:
                cells.groupby(grouping, as_index=False).agg(n=('speed_sq_error', 'size'),
                    speed_mse=('speed_sq_error', 'mean'), flow_lane_mse=('flow_lane_sq_error', 'mean')).to_csv(
                        destination / f'{method}_{name}_errors.csv', index=False)
        checks.extend(control_reference_checks(reference, pd.DataFrame(panel_rows), panel))
        save_json(output / 'reference_check.json', checks)
        if not all(c['passed'] for c in checks):
            raise ValueError('Source controls did not reproduce; inspect reference_check.json')
        rows.extend(panel_rows)
        raw_truth = day_files(root, panel, 'train', False)
        input_paths += list(paths.values()) + [folder / 'manifest.json']
        input_paths += [raw_truth[d] for d in manifest['metadata']['eval_dates']]
        reused.append(dict(panel=panel, evaluation_key=manifest['key'], evaluation_cache_hit=True,
            prediction_sources={m: str(p) for m, p in paths.items()}, target_rows=len(expected),
            gap_gate_direct_rows=int((~np.isfinite(diagnostics.gap_minutes) | (diagnostics.gap_minutes > 30)).sum()),
            retrained=False))
        detail = pd.DataFrame(rows)
        detail.to_csv(output / 'metrics_by_panel_regime.csv', index=False)
        aggregate(detail).to_csv(output / 'metrics_summary.csv', index=False)
        fusion_comparison(detail).to_csv(output / 'comparison_fusions.csv', index=False)
        save_json(output / 'prediction_reuse.json', reused)
        print(f'[{panel}] reused both saved predictions and verified cache; five methods scored', flush=True)
    save_json(output / 'input_manifest.json', snapshot(input_paths))
    save_json(output / 'COMPLETE.json', dict(experiment='multihop_fusion', panels=panels,
        source_run=str(source), controls=[RESIDUAL, DIRECT], rules=RULES, reference_passed=True, retrained=False))
    try:
        status = append_fusion_log(output, project / '实验日志.txt')
    except (OSError, ValueError) as error:
        save_json(output / 'log_update.json', dict(status='failed', error=str(error)))
        raise RuntimeError('Scoring completed; log append failed. Run log-fusion to retry without rescoring.') from error
    print(fusion_comparison(pd.DataFrame(rows)).to_string(index=False), flush=True)
    print(f'Log status: {status["status"]}. Fusion comparison complete: {output}', flush=True)
