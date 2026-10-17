import numpy as np
import pandas as pd
import pytest

from task1.data import HistoricalProfile
from task1.features import neighbor_map, temporal_context
from task1.metrics import aggregate, score, validate_predictions


def observation(value):
    return pd.DataFrame({'link_id': ['L1'], 'weekday': [0], 'slot': [0],
                         'pct_observed': [100], 'speed_kmh': [value], 'flow_vph': [value * 10]})


def test_profile_excludes_entire_training_day_and_restores_exactly():
    p = HistoricalProfile(['L1'])
    first, second = observation(50), observation(100)
    p.accumulate(first)
    p.accumulate(second)
    assert p.lookup(first)[0, 0] == 75
    p.accumulate(first, -1)
    assert p.lookup(first)[0, 0] == 100
    p.accumulate(first)
    assert p.lookup(first)[0, 0] == 75


def test_interpolation_uses_real_five_minute_spacing_and_preserves_long_gaps():
    x = np.full((288, 1), np.nan)
    x[0], x[4] = 20, 60
    prev, nxt, before, after, initial = temporal_context(x)
    assert initial[1, 0] == 30
    assert before[1, 0] == 5 and after[1, 0] == 15
    assert np.isnan(initial[100, 0])
    # The target cell itself is absent from strictly previous/next context.
    x[1] = 999
    p2, n2, *_ = temporal_context(x)
    assert p2[1, 0] == prev[1, 0] and n2[1, 0] == nxt[1, 0]


def test_topology_skips_uninstrumented_mainline_and_ignores_connectors():
    topology = pd.DataFrame({'mainline_link_id': ['A', 'B', 'C'],
                             'incoming_link_ids': ['', 'A', 'B;RAMP'],
                             'outgoing_link_ids': ['B;CONN', 'C', '']})
    m = neighbor_map(topology, {'A', 'C'})
    assert m[('A', 'down')] == ['C']
    assert m[('C', 'up')] == ['A']


def one_target():
    return pd.DataFrame([dict(panel='P', timestamp='2031-01-01T00:00:00Z',
                              station_id='S', link_id='L', mask_regime='R1',
                              speed_kmh=50., flow_vph=1000.)])


def test_flow_error_is_per_lane_and_coverage_is_strict():
    truth = one_target()
    pred = truth.copy()
    pred.flow_vph += 400
    net = pd.DataFrame({'lanes': [4.], 'free_speed_kmh': [100.]}, index=['L'])
    rows, _ = score(pred, truth, net, 'F', 'B0')
    assert rows[0]['rmse_flow_per_lane'] == 100
    assert rows[0]['S_state'] == pytest.approx(0.54 + 0.46 * (1 - 100 / 600))
    with pytest.raises(ValueError, match='coverage'):
        validate_predictions(truth, pred.iloc[:0])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_predictions(truth, pd.concat([pred, pred]))


def test_aggregation_is_regime_then_panel_then_family_equal_weight():
    details = pd.DataFrame([
        dict(method='B', family='F1', panel='P1', regime='R1', S_state=1., n=10000),
        dict(method='B', family='F1', panel='P1', regime='R2', S_state=0., n=1),
        dict(method='B', family='F1', panel='P1', regime='R3', S_state=0.5, n=50),
        dict(method='B', family='F1', panel='P2', regime='R1', S_state=0., n=5),
        dict(method='B', family='F2', panel='P3', regime='R1', S_state=1., n=5)])
    out = aggregate(details).iloc[0]
    assert out.S_state == pytest.approx(((0.5 + 0) / 2 + 1) / 2)
    assert not out.all_regimes_present


def test_model_roundtrip_in_chinese_project_path(tmp_path):
    import lightgbm as lgb
    from task1.pipeline import save_booster, load_booster
    folder = tmp_path / '比赛模型'
    folder.mkdir()
    x = np.arange(60, dtype=float).reshape(30, 2)
    model = lgb.train({'objective': 'regression', 'verbosity': -1, 'num_threads': 1,
                       'min_data_in_leaf': 2, 'min_data_in_bin': 1},
                      lgb.Dataset(x, label=x[:, 0] / 3), num_boost_round=3)
    save_booster(model, folder / 'speed.txt')
    loaded = load_booster(folder / 'speed.txt')
    np.testing.assert_allclose(model.predict(x), loaded.predict(x))
