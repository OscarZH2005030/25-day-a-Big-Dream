import json

import numpy as np
import pandas as pd
import pytest

from task1.data import KEY, TARGETS
from task1.experiment_log import append_fusion_log
from task1.fusion_experiments import (DIRECT, RESIDUAL, RULES, control_reference_checks,
                                      fuse_predictions, fusion_comparison)
from task1.metrics import score


def example_predictions():
    expected = pd.DataFrame(dict(panel=['P'] * 5, timestamp=['t1', 't2', 't3', 't4', 't5'],
        station_id=['S'] * 5, link_id=['L'] * 5, mask_regime=['R1'] * 5))
    r, d = expected.copy(), expected.copy()
    r[TARGETS] = [[10., 600.], [20., 900.], [30., 1200.], [40., 1500.], [50., 1800.]]
    d[TARGETS] = [[15., 900.], [25., 1200.], [35., 1500.], [45., 1800.], [55., 2100.]]
    return expected, r, d


def test_fusions_align_by_key_and_gate_boundaries_without_changing_total_flow_units():
    expected, r, d = example_predictions()
    outputs = fuse_predictions(expected, r.iloc[::-1], d.sample(frac=1, random_state=2),
                               [15., 30., 30.001, np.nan, np.inf])
    np.testing.assert_array_equal(outputs[RESIDUAL][TARGETS], r[TARGETS])
    np.testing.assert_array_equal(outputs[DIRECT][TARGETS], d[TARGETS])
    np.testing.assert_array_equal(outputs['Fusion_channels'].speed_kmh, r.speed_kmh)
    np.testing.assert_array_equal(outputs['Fusion_channels'].flow_vph, d.flow_vph)
    np.testing.assert_array_equal(outputs['Fusion_mean50'][TARGETS], .5 * (r[TARGETS] + d[TARGETS]))
    np.testing.assert_array_equal(outputs['Fusion_gap30'][TARGETS],
                                 pd.concat([r.iloc[:2], d.iloc[2:]])[TARGETS])
    for pred in outputs.values():
        pd.testing.assert_frame_equal(pred[KEY], expected)


def test_invalid_prediction_coverage_and_gap_are_rejected():
    expected, r, d = example_predictions()
    for broken in [d.iloc[:-1], pd.concat([d, d.iloc[:1]])]:
        with pytest.raises(ValueError):
            fuse_predictions(expected, r, broken, [15.] * 5)
    for gap in [[15.], [-1.] * 5, [0.] * 5]:
        with pytest.raises(ValueError, match='gap'):
            fuse_predictions(expected, r, d, gap)
    d.loc[0, 'flow_vph'] = np.nan
    with pytest.raises(ValueError, match='Non-finite'):
        fuse_predictions(expected, r, d, [15.] * 5)


def scored_example():
    expected, r, d = example_predictions()
    outputs = fuse_predictions(expected, r, d, [15., 30., 45., np.nan, np.inf])
    truth = expected.copy()
    truth[TARGETS] = [[12., 750.]] * 5
    net = pd.DataFrame({'lanes': [3], 'free_speed_kmh': [100.]}, index=['L'])
    rows = []
    for method, pred in outputs.items():
        detail, _ = score(pred, truth, net, 'family', method)
        rows.extend(detail)
    return pd.DataFrame(rows)


def test_comparison_uses_two_controls_and_official_lane_normalization():
    detail = scored_example()
    table = fusion_comparison(detail).set_index('method')
    assert table.loc[RESIDUAL, 'delta_S_state_vs_residual'] == 0
    assert table.loc[DIRECT, 'delta_S_state_vs_direct'] == 0
    np.testing.assert_allclose(table.loc['Fusion_channels', 'rmse_speed_kmh'], table.loc[RESIDUAL, 'rmse_speed_kmh'])
    np.testing.assert_allclose(table.loc['Fusion_channels', 'rmse_flow_per_lane'], table.loc[DIRECT, 'rmse_flow_per_lane'])
    np.testing.assert_allclose(table.loc['Fusion_channels', 'S_state'],
        .54 * (1 - table.loc[RESIDUAL, 'rmse_speed_kmh'] / 25) +
        .46 * (1 - table.loc[DIRECT, 'rmse_flow_per_lane'] / 600))
    assert all(c['passed'] for c in control_reference_checks(detail, detail, 'P'))
    changed = detail.copy()
    changed.loc[changed.method.eq(DIRECT), 'S_state'] += .001
    assert not all(c['passed'] for c in control_reference_checks(detail, changed, 'P'))


def completed_fusion(tmp_path):
    run = tmp_path / 'fusion_example'
    run.mkdir()
    cfg = dict(fit_start='2030-06-01', fit_end='2030-12-31', eval_start='2031-01-01',
               eval_end='2031-01-31', source_run='target_example', max_fit_days=None,
               max_eval_days=None, smoke=False, no_log=False)
    (run / 'run_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    (run / 'COMPLETE.json').write_text(json.dumps(dict(experiment='multihop_fusion',
        panels=['P'], controls=[RESIDUAL, DIRECT], rules=RULES, reference_passed=True)), encoding='utf-8')
    detail = scored_example()
    (run / 'reference_check.json').write_text(json.dumps(control_reference_checks(detail, detail, 'P')), encoding='utf-8')
    fusion_comparison(detail).to_csv(run / 'comparison_fusions.csv', index=False)
    return run, cfg


def test_fusion_logging_preserves_history_deduplicates_and_requires_complete_controls(tmp_path):
    run, _ = completed_fusion(tmp_path)
    log = tmp_path / 'log.txt'
    log.write_text('历史\n', encoding='utf-8-sig')
    assert append_fusion_log(run, log)['status'] == 'appended'
    first = log.read_text(encoding='utf-8-sig')
    assert first.startswith('历史\n') and all(m in first for m in [RESIDUAL, DIRECT, *RULES])
    assert append_fusion_log(run, log)['status'] == 'already_logged'
    assert log.read_text(encoding='utf-8-sig') == first
    (run / 'reference_check.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='every panel'):
        append_fusion_log(run, log)


@pytest.mark.parametrize('option,value,status', [('smoke', True, 'skipped_smoke'),
    ('max_fit_days', 14, 'skipped_smoke'), ('max_eval_days', 3, 'skipped_smoke'),
    ('no_log', True, 'skipped_by_option')])
def test_fusion_smoke_and_optout_do_not_pollute_log(tmp_path, option, value, status):
    run, cfg = completed_fusion(tmp_path)
    cfg[option] = value
    (run / 'run_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    log = tmp_path / 'log.txt'
    assert append_fusion_log(run, log)['status'] == status
    assert not log.exists()
