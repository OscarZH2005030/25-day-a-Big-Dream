import json
import os
import uuid

import pandas as pd
import pytest

from resume_task1_baseline import CachePublishError, RecoverableCache, append_log
from task1.cache import CacheIntegrityError, cache_key
from task1.data import save_json, sha256
from task1.corridor_validation import METHODS


def completed_temporary(root, identity=None):
    identity = identity or dict(kind='train', panel='P', seed=1)
    folder = root / ('.tmp_' + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    (folder / 'features.txt').write_text('complete features', encoding='utf-8')
    save_json(folder / 'manifest.json', dict(key=cache_key(identity), identity=identity,
        metadata={'rows': 3}, artifacts=[dict(name='features.txt', sha256=sha256(folder / 'features.txt'))]))
    return folder, identity


def test_denied_native_rename_uses_fallback_and_preserves_content(tmp_path):
    temporary, identity = completed_temporary(tmp_path)
    calls = []
    def denied(*args):
        raise PermissionError('Windows access denied')
    def fallback(source, destination):
        calls.append((source, destination))
        os.rename(source, destination)
    cache = RecoverableCache(tmp_path, native_replace=denied, rename_fallback=fallback)
    folder, _, hit = cache.publish(temporary, identity)
    assert not hit and len(calls) == 1 and not temporary.exists()
    assert (folder / 'features.txt').read_text(encoding='utf-8') == 'complete features'


def test_persistent_failure_keeps_complete_cache_then_next_run_recovers_without_build(tmp_path):
    temporary, identity = completed_temporary(tmp_path)
    calls = []
    def denied(*args):
        calls.append(1)
        raise PermissionError('still locked')
    cache = RecoverableCache(tmp_path, retries=2, native_replace=denied, rename_fallback=denied, sleep=lambda _: None)
    with pytest.raises(CachePublishError, match='complete and preserved'):
        cache.publish(temporary, identity)
    assert temporary.exists() and len(calls) == 4
    cache._validate(temporary, identity)
    cache = RecoverableCache(tmp_path)
    def must_not_build(_):
        pytest.fail('Completed cache must not be recomputed')
    folder, _, hit = cache.get_or_build(identity, must_not_build)
    assert hit and folder.name == cache_key(identity)


def test_readonly_scan_then_recovery_skips_incomplete_and_keeps_corrupt_entries(tmp_path):
    valid, identity = completed_temporary(tmp_path)
    broken, _ = completed_temporary(tmp_path, dict(kind='train', panel='P', seed=2))
    (broken / 'features.txt').write_text('corrupt', encoding='utf-8')
    incomplete = tmp_path / ('.tmp_' + uuid.uuid4().hex)
    incomplete.mkdir()
    (incomplete / 'partial.txt').write_text('not done', encoding='utf-8')
    cache = RecoverableCache(tmp_path)
    records = cache.recover_completed(publish=False, panels={'P'})
    assert {r['status'] for r in records} == {'verified_ready', 'invalid_kept'}
    assert valid.exists() and not (tmp_path / cache_key(identity)).exists()
    records = cache.recover_completed(publish=True, panels={'P'})
    assert any(r['status'] == 'recovered' for r in records)
    assert broken.exists() and incomplete.exists()


def test_existing_valid_destination_is_not_overwritten(tmp_path):
    first, identity = completed_temporary(tmp_path)
    second, _ = completed_temporary(tmp_path, identity)
    cache = RecoverableCache(tmp_path)
    destination, _, _ = cache.publish(first, identity)
    before = sha256(destination / 'manifest.json')
    folder, _, hit = cache.publish(second, identity)
    assert hit and folder == destination and second.exists()
    assert sha256(destination / 'manifest.json') == before


def test_existing_corrupt_destination_is_not_overwritten(tmp_path):
    temporary, identity = completed_temporary(tmp_path)
    destination = tmp_path / cache_key(identity)
    destination.mkdir()
    with pytest.raises(CacheIntegrityError):
        RecoverableCache(tmp_path).publish(temporary, identity)
    assert temporary.exists()


def test_publication_rejects_paths_outside_cache_root(tmp_path):
    temporary, identity = completed_temporary(tmp_path / 'elsewhere')
    cache = RecoverableCache(tmp_path / 'cache')
    with pytest.raises(ValueError, match='outside'):
        cache.publish(temporary, identity)
    assert temporary.exists()


def test_new_cache_keeps_original_identity_and_warm_hit_skips_builder(tmp_path):
    cache = RecoverableCache(tmp_path)
    identity = dict(kind='train', panel='P')
    calls = []
    def builder(folder):
        calls.append(1)
        (folder / 'data.txt').write_text('x', encoding='utf-8')
        return {'rows': 1}
    first = cache.get_or_build(identity, builder)
    second = cache.get_or_build(identity, builder)
    assert not first[2] and second[2] and calls == [1]
    assert first[0].name == cache_key(identity)
    assert first[1]['identity'] == identity


def test_complete_all_panel_log_preserves_history_and_deduplicates(tmp_path):
    run = tmp_path / 'run'; run.mkdir()
    panels = [f'P{i}' for i in range(10)]
    save_json(run / 'COMPLETE.json', dict(experiment='all_task1_baseline', panels=panels))
    save_json(run / 'run_config.json', dict(fit_start='2030-06-01', fit_end='2030-12-31',
        eval_start='2031-01-01', eval_end='2031-01-31', sample_per_day=1000,
        rounds=350, seed=20261016, threads=4))
    rows = [dict(method=m, S_state=.9, panel_count=10, family_count=5, all_regimes_present=True,
        rmse_speed_kmh=2., rmse_flow_per_lane=40., delta_S_state_vs_residual=.001,
        delta_S_state_vs_direct=.002, delta_S_state_vs_mean50=0.) for m in METHODS]
    pd.DataFrame(rows).to_csv(run / 'comparison_overall.csv', index=False)
    pd.DataFrame([dict(panel=p, **r) for p in panels for r in rows]).to_csv(run / 'comparison_by_panel.csv', index=False)
    log = tmp_path / 'log.txt'; log.write_text('历史\n', encoding='utf-8-sig')
    assert append_log(run, log)['status'] == 'appended'
    first = log.read_text(encoding='utf-8-sig')
    assert first.startswith('历史\n') and all(m in first for m in METHODS)
    assert append_log(run, log)['status'] == 'already_logged'
    assert log.read_text(encoding='utf-8-sig') == first
    pd.DataFrame(rows[:-1]).to_csv(run / 'comparison_overall.csv', index=False)
    with pytest.raises(ValueError, match='Incomplete overall'):
        append_log(run, log)
