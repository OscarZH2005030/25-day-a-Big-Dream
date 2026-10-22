import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from task1.cache_builder import IDENTIFIER_COLUMNS, RESIDUAL_COLUMNS
from task1.data import KEY
from task1.experiment_log import append_target_log
from task1.experiments import compose_target_prediction, fit_cached
from task1.pipeline import load_booster
from task1.target_labels import DIRECT_TARGET_COLUMNS, aligned_direct_targets, exact_day_targets


def test_prediction_mode_adds_initial_only_for_residual():
    initial = np.array([[100., 500.]])
    prediction = np.array([[5., 10.]])
    np.testing.assert_array_equal(compose_target_prediction(initial, prediction, 'residual'), [[105., 510.]])
    np.testing.assert_array_equal(compose_target_prediction(initial, prediction, 'direct'), prediction)
    with pytest.raises(ValueError):
        compose_target_prediction(initial, prediction, 'typo')


def test_exact_direct_labels_join_keys_preserve_float64_and_lane_units():
    keys = pd.DataFrame({'timestamp': ['t2', 't1'], 'station_id': ['S2', 'S1'], 'link_id': ['L2', 'L1']})
    complete = pd.DataFrame({'timestamp': ['t1', 't2'], 'station_id': ['S1', 'S2'],
        'link_id': ['L1', 'L2'], 'speed_kmh': [123.123456789123, 87.987654321987],
        'flow_vph': [5678.90123456789, 4321.98765432198], 'pct_observed': [100, 75]})
    net = pd.DataFrame({'lanes': [3., 4.]}, index=['L1', 'L2'])
    labels = exact_day_targets(keys, complete, net)
    expected = np.array([[complete.speed_kmh.iloc[1], complete.flow_vph.iloc[1] / 4],
                         [complete.speed_kmh.iloc[0], complete.flow_vph.iloc[0] / 3]])
    np.testing.assert_array_equal(labels, expected)
    assert labels.dtype == np.float64
    assert labels[0, 0] != np.float64(np.float32(labels[0, 0]))
    complete.loc[0, 'pct_observed'] = 0
    with pytest.raises(ValueError, match='unavailable'):
        exact_day_targets(keys, complete, net)


def tiny_training_table():
    n = 30
    keys = pd.DataFrame({'panel': ['P'] * n,
        'timestamp': pd.date_range('2030-06-01', periods=n, freq='5min', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'),
        'station_id': ['S'] * n, 'link_id': ['L'] * n, 'mask_regime': ['R1'] * n})
    x = pd.DataFrame({'speed_initial': np.arange(n, dtype=np.float32) + 50,
                       'flow_lane_initial': np.arange(n, dtype=np.float32) + 300})
    x[IDENTIFIER_COLUMNS] = keys[KEY].to_numpy()
    x[RESIDUAL_COLUMNS] = np.column_stack([np.arange(n) / 10, np.arange(n) / 5])
    labels = x[IDENTIFIER_COLUMNS].copy()
    labels[DIRECT_TARGET_COLUMNS] = np.column_stack([np.arange(n) + 100., np.arange(n) + 900.])
    return x, labels


def test_direct_label_reordering_and_nonfinite_values_rejected():
    training, labels = tiny_training_table()
    aligned_direct_targets(training, labels)
    with pytest.raises(ValueError, match='IDs/order'):
        aligned_direct_targets(training, labels.iloc[::-1])
    labels.loc[0, DIRECT_TARGET_COLUMNS[0]] = np.nan
    with pytest.raises(ValueError, match='Non-finite'):
        aligned_direct_targets(training, labels)


def test_paired_fitting_shares_features_but_uses_correct_target(tmp_path):
    training, labels = tiny_training_table()
    cache = tmp_path / 'cache'
    cache.mkdir()
    training.to_parquet(cache / 'training.parquet', index=False)
    labels.to_parquet(cache / 'direct_targets.parquet', index=False)
    (cache / 'profile.pkl').write_bytes(b'local-test-profile')
    manifest = dict(key='test', metadata=dict(feature_columns=['speed_initial', 'flow_lane_initial'],
                    fit_dates=['2030-06-01']))
    cfg = dict(seed=1, threads=1, rounds=3)
    for mode in ['residual', 'direct']:
        fit_cached(cache, manifest, tmp_path / mode, cfg, 'baseline', target_mode=mode,
                   label_folder=cache if mode == 'direct' else None, metadata_name='target_model_config.json')
    a = json.loads((tmp_path / 'residual/target_model_config.json').read_text())
    b = json.loads((tmp_path / 'direct/target_model_config.json').read_text())
    assert a['features'] == b['features'] and a['parameters'] == b['parameters']
    assert a['training_table_sha256'] == b['training_table_sha256']
    assert not (tmp_path / 'direct/model_config.json').exists()
    pred = load_booster(tmp_path / 'direct/speed.txt').predict(training[a['features']])
    # min_data_in_leaf=50 and 30 rows: constant fit must equal direct-target mean.
    np.testing.assert_allclose(pred, labels[DIRECT_TARGET_COLUMNS[0]].mean())


def make_completed_log_run(tmp_path, smoke=False):
    run = tmp_path / 'target_example'
    run.mkdir()
    cfg = dict(fit_start='2030-06-01', fit_end='2030-12-31', eval_start='2031-01-01',
               eval_end='2031-01-31', sample_per_day=1000, rounds=350, threads=4, seed=1,
               max_fit_days=14 if smoke else None, max_eval_days=3 if smoke else None,
               reference_run=None, no_log=False)
    (run / 'run_config.json').write_text(json.dumps(cfg), encoding='utf-8')
    (run / 'COMPLETE.json').write_text(json.dumps(dict(experiment='target_comparison', panels=['P'])), encoding='utf-8')
    pd.DataFrame([dict(feature_variant=v, n_features=50, residual_S_state=.9, direct_S_state=.91,
       delta_direct_minus_residual=.01, residual_speed_rmse=2., direct_speed_rmse=1.9,
       residual_flow_rmse=40., direct_flow_rmse=39., winner='direct')
       for v in ['baseline', 'multihop', 'congestion']]).to_csv(run / 'comparison_target_modes.csv', index=False)
    return run


def test_log_is_appended_once_and_preserves_existing_text(tmp_path):
    run = make_completed_log_run(tmp_path)
    log = tmp_path / '实验日志.txt'
    log.write_text('已有历史记录\n', encoding='utf-8-sig')
    assert append_target_log(run, log)['status'] == 'appended'
    first = log.read_text(encoding='utf-8-sig')
    assert '已有历史记录' in first and '直接：S_state=0.910000' in first
    assert append_target_log(run, log)['status'] == 'already_logged'
    assert log.read_text(encoding='utf-8-sig') == first


def test_smoke_and_incomplete_runs_do_not_pollute_formal_log(tmp_path):
    run = make_completed_log_run(tmp_path, smoke=True)
    log = tmp_path / 'log.txt'
    assert append_target_log(run, log)['status'] == 'skipped_smoke'
    assert not log.exists()
    with pytest.raises(FileNotFoundError):
        append_target_log(tmp_path / 'incomplete', log)
