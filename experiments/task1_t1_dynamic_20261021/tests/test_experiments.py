import json

import numpy as np
import pandas as pd
import pytest

from task1.cache import CacheIntegrityError, FeatureCache, cache_key, snapshot
from task1.data import HistoricalProfile
from task1.features import (build_features, congestion_features, feature_columns,
                            multihop_neighbors)


def fixture_day():
    links = ['A', 'B', 'C', 'D']
    topology = pd.DataFrame({'mainline_link_id': links,
        'incoming_link_ids': ['', 'A', 'B', 'C'],
        'outgoing_link_ids': ['B', 'C', 'D', '']})
    net = pd.DataFrame({'length_km': [1., 2., 1., 2.], 'lanes': [2.] * 4,
                       'free_speed_kmh': [100.] * 4, 'capacity_vph': [4000.] * 4}, index=links)
    rows = []
    for i, link in enumerate(links):
        for slot in range(288):
            rows.append(dict(link_id=link, station_id='S' + link, slot=slot, weekday=0,
                             pct_observed=100, speed_kmh=50. + i + slot / 100,
                             flow_vph=1000. + slot))
    f = pd.DataFrame(rows)
    profile = HistoricalProfile(links)
    profile.accumulate(f)
    f.loc[(f.link_id == 'B') & f.slot.between(100, 102), ['speed_kmh', 'flow_vph']] = np.nan
    return f, profile, net, topology


def test_two_variants_are_isolated_and_baseline_columns_unchanged():
    f, profile, net, topology = fixture_day()
    base, initial, history, lanes = build_features(f, profile, net, topology)
    all_x, initial2, history2, lanes2 = build_features(f, profile, net, topology, feature_set='all')
    assert len(base.columns) == 50
    pd.testing.assert_frame_equal(base, all_x[feature_columns(all_x.columns, 'baseline')])
    np.testing.assert_array_equal(initial, initial2)
    np.testing.assert_array_equal(history, history2)
    np.testing.assert_array_equal(lanes, lanes2)
    mh = feature_columns(all_x.columns, 'multihop')
    cg = feature_columns(all_x.columns, 'congestion')
    assert any(c.startswith('mh_') for c in mh) and not any(c.startswith('cg_') for c in mh)
    assert any(c.startswith('cg_') for c in cg) and not any(c.startswith('mh_') for c in cg)
    assert set(mh).intersection(cg) == set(base.columns)
    assert np.isnan(all_x.loc[0, 'cg_speed_past_slope_per_min'])


def test_multihop_distance_skips_unobserved_links_and_handles_cycles():
    topology = pd.DataFrame({'mainline_link_id': ['A', 'X', 'B', 'C', 'D'],
        'incoming_link_ids': ['', 'A', 'X', 'B', 'C'],
        'outgoing_link_ids': ['X;CONN', 'B', 'C', 'D', 'A']})
    net = pd.DataFrame({'length_km': [1., 2., 1., 2., 1.]}, index=['A', 'X', 'B', 'C', 'D'])
    result = multihop_neighbors(topology, {'A', 'B', 'C', 'D'}, net)
    assert result[('A', 'down', 1)] == [('B', 3.)]
    assert result[('A', 'down', 2)] == [('C', 4.5)]
    assert result[('A', 'down', 3)] == [('D', 6.)]
    assert all(origin != neighbor for (origin, _, _), pairs in result.items() for neighbor, _ in pairs)


def test_cache_hit_skips_builder_and_corruption_is_rejected(tmp_path):
    cache = FeatureCache(tmp_path / 'cache')
    identity = {'kind': 'train', 'panel': 'P', 'dates': ['2030-06-01'], 'seed': 1}
    calls = []
    def builder(folder):
        calls.append(1)
        (folder / 'features.txt').write_text('features', encoding='utf-8')
        return {'rows': 3}
    first, manifest, hit = cache.get_or_build(identity, builder)
    assert not hit
    second, m2, hit = cache.get_or_build(identity, builder)
    assert hit and first == second and manifest == m2 and len(calls) == 1
    (first / 'features.txt').write_text('changed', encoding='utf-8')
    with pytest.raises(CacheIntegrityError):
        cache.get_or_build(identity, builder)


def test_cache_identity_changes_with_data_dates_seed_and_source(tmp_path):
    p = tmp_path / 'input.txt'
    p.write_text('a', encoding='utf-8')
    original = dict(kind='train', panel='P', dates=['2030-06-01'], seed=1,
                    source='version1', inputs=snapshot([p]))
    first = cache_key(original)
    for field, value in [('dates', ['2030-06-02']), ('seed', 2), ('source', 'version2')]:
        assert cache_key({**original, field: value}) != first
    # Same byte count: content changes must still invalidate the identity.
    p.write_text('b', encoding='utf-8')
    assert cache_key({**original, 'inputs': snapshot([p])}) != first


def test_incomplete_cache_is_not_reused(tmp_path):
    cache = FeatureCache(tmp_path)
    identity = dict(kind='train', panel='P')
    (tmp_path / cache_key(identity)).mkdir()
    with pytest.raises(CacheIntegrityError, match='Incomplete'):
        cache.get_or_build(identity, lambda folder: {})
