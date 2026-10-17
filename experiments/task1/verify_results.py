"""Verify the published aggregates and independent SSE evidence; standard library only.

Does not import estimators, read datasets, train, initialize CUDA, or write files.
"""
from collections import defaultdict
from pathlib import Path
import ast
import csv
import hashlib
import json
import math

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO/'docs/results/task1_time80_20'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def near(a, b, label):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f'{label}: {a} != {b}')


def state(speed, flow):
    return .54*max(0., 1-speed/25.)+.46*max(0., 1-flow/600.)


def main():
    provenance = read(RESULTS/'provenance.json')
    for entry in provenance['sources']+provenance['immutable_artifacts']:
        path = REPO/entry['path']
        assert path.resolve().is_relative_to(REPO.resolve())
        digest = entry.get('repository_sha256', entry.get('sha256'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, entry['path']
    for path in (REPO/'experiments/task1').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    plan = read(RESULTS/'split.json')
    splits = [plan[key] for key in ['history_dates', 'training_dates', 'internal_validation_dates', 'holdout_dates']]
    assert [len(dates) for dates in splits] == [28, 168, 22, 55]
    all_dates = sum(splits, [])
    assert all_dates == sorted(set(all_dates)) and len(all_dates) == 273
    assert splits[0][0] == '2030-06-01' and splits[-1][-1] == '2031-02-28'
    panels = plan['panels']
    assert len(panels) == len(set(panels)) == 10
    assert len(read(RESULTS/'feature_names.json')) == 51
    holdout = RESULTS/'holdout_isolated_fallback'
    regime_rows = rows(holdout/'regime_metrics.csv')
    index = {(row['panel'], row['regime'], row['method']): row for row in regime_rows}
    assert len(index) == len(regime_rows) == 150
    panel_scores = defaultdict(list)
    method_scores = defaultdict(list)
    method_errors = defaultdict(lambda: [[], []])
    for row in regime_rows:
        speed, flow = float(row['rmse_speed_kmh']), float(row['rmse_flow_per_lane'])
        assert math.isfinite(speed) and math.isfinite(flow) and min(speed, flow) >= 0
        near(row['S_speed'], max(0., 1-speed/25.), 'speed score')
        near(row['S_flow'], max(0., 1-flow/600.), 'flow score')
        near(row['S_state'], state(speed, flow), 'state formula')
        panel_scores[row['panel'], row['method']].append(float(row['S_state']))
        method_scores[row['method']].append(float(row['S_state']))
        method_errors[row['method']][0].append(speed)
        method_errors[row['method']][1].append(flow)
    for row in rows(holdout/'method_summary.csv'):
        method = row['method']
        near(row['S_state'], math.fsum(method_scores[method])/30, 'equal corridor/regime mean')
        near(row['rmse_speed_kmh'], math.fsum(method_errors[method][0])/30, 'macro speed RMSE')
        near(row['rmse_flow_per_lane'], math.fsum(method_errors[method][1])/30, 'macro flow RMSE')
    for row in rows(holdout/'ten_corridor_comparison.csv'):
        for method in ['LightGBM', 'Transformer', 'isolated_both']:
            near(row['S_state__'+method], math.fsum(panel_scores[row['panel'], method])/3, 'corridor mean')
    # Reconstruct each model and routed channel directly from earlier structure SSE.
    evidence = defaultdict(list)
    for row in rows(RESULTS/'neighbor_failure_analysis/structure_errors_by_regime.csv'):
        evidence[row['panel'], row['regime']].append(row)
    for (panel, regime), group in evidence.items():
        total = sum(int(row['n']) for row in group)
        for method in ['LightGBM', 'Transformer', 'isolated_both']:
            sums = [0., 0.]
            for row in group:
                source = 'lgb' if method == 'LightGBM' or (method == 'isolated_both' and row['structure'] == 'isolated') else 'tf'
                for channel, name in enumerate(['speed', 'flow']):
                    sums[channel] += float(row[source+'_'+name+'_sse'])
            expected = [math.sqrt(value/total) for value in sums]
            actual = index[panel, regime, method]
            assert int(actual['n']) == total
            near(actual['rmse_speed_kmh'], expected[0], 'independent speed SSE')
            near(actual['rmse_flow_per_lane'], expected[1], 'independent flow SSE')
            near(actual['S_state'], state(*expected), 'independent state SSE')
    assert sum(int(row['n']) for row in regime_rows if row['method'] == 'LightGBM') == 6636705
    checks = read(holdout/'checks.json')
    assert checks['fallback_targets'] == 1469797
    # Original per-model exports must agree with the paired evaluation, not just
    # with another copy of the same summary table.
    for folder, method in [('transformer', 'Transformer'), ('lightgbm', 'LightGBM')]:
        original = rows(RESULTS/folder/'all_metrics.csv')
        assert len(original) == 40
        for row in original:
            key = row['panel'], method
            if row['regime'] == 'macro_mean':
                near(row['S_state'], math.fsum(panel_scores[key])/3, 'original corridor score')
            else:
                actual = index[row['panel'], row['regime'], method]
                assert int(row['n']) == int(actual['n'])
                for field in ['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']:
                    near(row[field], actual[field], 'original paired '+field)
    print('PASS: source/artifact SHA256, Python syntax, chronological splits, 51 features,')
    print('150 regime rows, ten-corridor/channel means, 6,636,705 targets, 1,469,797 fallback,')
    print('and independent structure-SSE reconstruction. No training, inference or CUDA.')


if __name__ == '__main__':
    main()
