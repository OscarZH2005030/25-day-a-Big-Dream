"""Portable references and adapters around the unchanged experiment engine."""
from pathlib import Path
import importlib.util

import pandas as pd

from task1.cache import snapshot
from task1.data import sha256, save_json
from task1.transformer_priority.common import read
from transformer_remaining import common
from transformer_remaining import queue as original_queue
from transformer_remaining import summary as original_summary

PORTABLE_EXPERIMENT = 'remaining_t1_dynamic_portable_queue_v1'
EXPECTED_REFERENCES = {
    'd7': ['D7_I405_N', 'D7_I405_S'],
    'i5': ['D12_I5_N', 'D12_I5_S'],
}


def reference_manifest(project):
    folder = Path(project).resolve() / 'references'
    manifest = read(folder / 'manifest.json')
    if manifest.get('schema') != 'frozen_four_direction_metric_archive_v1':
        raise ValueError('Unknown historical reference archive')
    return folder, manifest


def verify_reference_archive(project):
    """Verify copied tables using their original, completed-result manifests.

    This validates the exported metrics. It does not pretend to revalidate the
    original model weights or per-target predictions on the receiving machine.
    """
    folder, manifest = reference_manifest(project)
    artifacts = manifest['files']
    paths = [item['path'] for item in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate archive artifact')
    for item in artifacts:
        path = (folder / item['path']).resolve()
        if not path.is_relative_to(folder) or sha256(path) != item['sha256']:
            raise ValueError('Historical reference artifact changed: ' + item['path'])
    sources = manifest['sources']
    if len(sources) != 2 or {s['kind'] for s in sources} != set(EXPECTED_REFERENCES):
        raise ValueError('Both completed four-direction sources are required')
    reference = pd.read_csv(Path(project) / 'task1/transformer_priority/reference_routing_counts.csv').set_index('panel')
    specs = []
    for source in sources:
        path = (folder / source['folder']).resolve()
        if not path.is_relative_to(folder) or source['panels'] != EXPECTED_REFERENCES[source['kind']]:
            raise ValueError('Historical source scope changed')
        spec = dict(kind=source['kind'], path=str(path), panels=source['panels'])
        cfg = read(path / 'run_config.json')
        original_summary.validate_source_config(cfg, spec)
        complete = read(path / 'source_COMPLETE.json')
        saved_path = path / 'source_artifact_manifest.json'
        if (not complete['integrity_passed']
                or sha256(path / 'source_COMPLETE.json') != source['completion_sha256']
                or sha256(saved_path) != source['artifact_manifest_sha256']
                or complete['artifact_manifest_sha256'] != source['artifact_manifest_sha256']):
            raise ValueError('Historical completion proof changed')
        saved = {item['path']: item['sha256'] for item in read(saved_path)}
        for artifact in artifacts:
            if artifact['source_kind'] != source['kind']:
                continue
            if artifact['path'].endswith(('/source_COMPLETE.json', '/source_artifact_manifest.json')):
                continue
            if saved.get(artifact['original_path']) != artifact['sha256']:
                raise ValueError('Historical export is not in the completed source manifest')
        original_summary.validate_tables(
            pd.read_csv(path / 'holdout_metrics_by_panel_regime.csv'),
            pd.read_csv(path / 'holdout_daily_errors.csv'),
            read(path / 'holdout_preservation_checks.json'), spec, reference)
        specs.append(spec)
    return specs


def _isolated(module, name):
    spec = importlib.util.spec_from_file_location(name, Path(module.__file__))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def portable_queue(project):
    """Reuse the original queue, batch engines, scoring, resume and logging.

    Only the four already-completed directions' source adapter changes: they
    are read from the verified table archive shipped alongside the source.
    """
    project = Path(project).resolve()
    queue = _isolated(original_queue, 'transformer_remaining._portable_queue')
    summary = _isolated(original_summary, 'transformer_remaining._portable_summary')

    def verify_source(spec):
        if spec['kind'] == 'batch':
            return original_summary.verify_source(spec)
        expected = verify_reference_archive(project)
        if spec not in expected:
            raise ValueError('Reference source is outside the frozen archive')
        return read(Path(spec['path']) / 'run_config.json')

    def anchors(specs):
        archived = [spec for spec in specs if spec['kind'] != 'batch']
        batches = [spec for spec in specs if spec['kind'] == 'batch']
        records = []
        if archived:
            expected = verify_reference_archive(project)
            if any(spec not in expected for spec in archived):
                raise ValueError('Unrecognized historical reference')
            folder, manifest = reference_manifest(project)
            kinds = {spec['kind'] for spec in archived}
            records += snapshot([folder / 'manifest.json'] + [folder / a['path'] for a in manifest['files'] if a['source_kind'] in kinds])
        if batches:
            records += original_summary.anchor_sources(batches)
        return records

    summary.verify_source = verify_source
    summary.anchor_sources = anchors

    def summarize(out, root, cfg):
        table = summary.summarize_queue(out, root, cfg)
        integrity = read(out / 'summary_integrity.json')
        integrity.update(
            historical_reference_mode='verified frozen metric archive',
            newly_trained_directions=common.PANELS,
            reused_historical_directions=sum(EXPECTED_REFERENCES.values(), []),
            historical_weights_revalidated_on_this_computer=False)
        save_json(out / 'summary_integrity.json', integrity)
        return table

    def dependencies():
        extra = ['team_remaining.py', 'team_portable.py', 'check_environment.py',
                 'run_team_remaining.ps1', 'setup_team.ps1', 'requirements-lock.txt',
                 'requirements-torch-deps.txt', 'source_export_manifest.json']
        return common.dependencies() + snapshot(project / name for name in extra)

    queue.QUEUE_EXPERIMENT = PORTABLE_EXPERIMENT
    queue.old_sources = verify_reference_archive
    queue.anchor_sources = anchors
    queue.verify_source = verify_source
    queue.summarize_queue = summarize
    queue.dependencies = dependencies
    return queue
