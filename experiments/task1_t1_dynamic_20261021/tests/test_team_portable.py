from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import pandas as pd
import pytest

from check_environment import verify_sources
from task1.data import save_json
from task1.transformer_priority.common import versions
from transformer_remaining import common, queue as original_queue, summary as original_summary
from team_portable import portable_queue, verify_reference_archive, PORTABLE_EXPERIMENT
from team_remaining import arguments

ROOT = Path(__file__).resolve().parents[1]


def test_training_sources_are_byte_identical_to_running_experiment():
    assert verify_sources() == 107


def test_archive_validates_original_results_and_uses_only_local_paths():
    sources = verify_reference_archive(ROOT)
    assert {p for spec in sources for p in spec['panels']} == {'D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S'}
    assert all(Path(spec['path']).is_relative_to(ROOT / 'references') for spec in sources)


def test_changed_reference_metrics_are_rejected(tmp_path):
    shutil.copytree(ROOT / 'references', tmp_path / 'references')
    shutil.copytree(ROOT / 'task1/transformer_priority', tmp_path / 'task1/transformer_priority')
    path = tmp_path / 'references/d7_i405/holdout_metrics_by_panel_regime.csv'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='artifact changed'):
        verify_reference_archive(tmp_path)


def config_args(tmp_path):
    return SimpleNamespace(action='run', project=ROOT, release_root=tmp_path/'official_data',
        cache_dir=tmp_path/'cache', output=tmp_path/'output', device='cpu', smoke=True,
        batches=common.BATCH_NAMES.copy(), resume=False)


def test_fresh_computer_config_needs_no_historical_runs_or_caches(tmp_path):
    queue = portable_queue(ROOT)
    cfg = queue.build_config(config_args(tmp_path))
    queue.validate_queue_config(cfg)
    assert cfg['experiment'] == PORTABLE_EXPERIMENT
    assert all('/references/' in Path(s['path']).as_posix() for s in cfg['old_sources'])
    assert cfg['training'] == dict(common.CONFIG, epochs=2, steps_per_epoch=4, patience=2)
    assert cfg['fixed_fusion'] == dict(beta=.5, speed=.1, flow=.1)
    assert not (tmp_path/'cache').exists() and not (tmp_path/'output').exists()
    assert original_queue.QUEUE_EXPERIMENT != PORTABLE_EXPERIMENT
    assert queue.run_queue.__globals__ is not original_queue.run_queue.__globals__


def test_new_entry_requires_top_level_output_to_resume():
    with pytest.raises(SystemExit):
        arguments(['run','--resume'])


def test_portable_queue_failure_and_resume_keep_completed_batches(monkeypatch, tmp_path):
    queue = portable_queue(ROOT)
    args = config_args(tmp_path)
    monkeypatch.setattr(queue, 'inspect_child', lambda *a, **k: None)
    monkeypatch.setattr(queue, 'summarize_smoke', lambda *a: None)
    calls = []
    failed = False
    def launch(args, batch, folder, resume):
        nonlocal failed
        calls.append((batch['name'], resume))
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder/'run_config.json', {})
        if batch['name'] == 'i10' and not failed:
            failed = True
            raise RuntimeError('injected child failure')
        save_json(folder/'COMPLETE.json', {})
    monkeypatch.setattr(queue, 'spawn_child', launch)
    with pytest.raises(RuntimeError, match='injected'):
        queue.run_queue(args)
    assert calls == [('i210', False), ('i10', False)]
    args.resume = True
    assert queue.run_queue(args) == 0
    assert calls == [('i210', False), ('i10', False), ('i10', True), ('i405', False)]
    assert (args.output/'SMOKE_COMPLETE.json').is_file()
    assert not (args.output/'COMPLETE.json').exists()


def test_missing_archive_fails_before_creating_output(tmp_path):
    args = config_args(tmp_path)
    args.project = tmp_path/'empty_checkout'
    queue = portable_queue(args.project)
    with pytest.raises(FileNotFoundError):
        queue.run_queue(args)
    assert not args.output.exists()


def test_real_four_direction_archive_combines_with_six_new_sources(monkeypatch, tmp_path):
    from test_transformer_remaining import synthetic_ten
    detail, daily, checks, ref = synthetic_ten()
    args = config_args(tmp_path)
    args.smoke = False
    queue = portable_queue(ROOT)
    cfg = queue.build_config(args)
    for batch in cfg['batches']:
        folder = args.output/'batches'/batch['folder']
        folder.mkdir(parents=True)
        panels = batch['panels']
        detail.loc[detail.panel.isin(panels)].to_csv(folder/'holdout_metrics_by_panel_regime.csv', index=False)
        daily.loc[daily.panel.isin(panels)].to_csv(folder/'holdout_daily_errors.csv', index=False)
        save_json(folder/'holdout_preservation_checks.json', checks)
        for panel in panels:
            (folder/'holdout'/panel).mkdir(parents=True)
            for method in common.SUMMARY_METHODS:
                for kind in ['gap','flow_gap','regime_gap','congestion','isolation']:
                    pd.DataFrame([dict(panel=panel, method=method, n=int(ref.loc[panel,'n']))]).to_csv(
                        folder/'holdout'/panel/f'{method}_{kind}_errors.csv', index=False)
    monkeypatch.setattr(original_summary, 'verify_source', lambda spec: dict(versions=versions()))
    monkeypatch.setattr(original_summary, 'anchor_sources', lambda specs: [])
    table = queue.summarize_queue(args.output, ROOT, cfg)
    combined = pd.read_csv(args.output/'ten_direction_metrics_by_panel_regime.csv')
    assert set(combined.panel) == set(ref.index)
    for row in table.itertuples(index=False):
        expected = combined.loc[combined.method.eq(row.method)].groupby(['family','panel']).S_state.mean().groupby('family').mean().mean()
        assert row.S_state == pytest.approx(expected, abs=1e-13)
    integrity = json.loads((args.output/'summary_integrity.json').read_text(encoding='utf-8'))
    assert integrity['panels'] == 10
    assert integrity['historical_weights_revalidated_on_this_computer'] is False
    assert integrity['historical_reference_mode'] == 'verified frozen metric archive'
