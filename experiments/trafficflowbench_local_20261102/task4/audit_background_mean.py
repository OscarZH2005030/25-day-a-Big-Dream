"""Reproduce public-input background-mean checks without local research caches."""
import argparse
import shutil
import sys
import types
from pathlib import Path
from datetime import datetime, timezone
from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parents[1]
    archive = here / 'archive/daytime'
    common = types.ModuleType('common')
    common.DATA = args.release_root.resolve()
    common.WORK = common.ROOT = work
    common.OUT = out
    common.sha, common.save = sha, save
    common.check_deadline = lambda: None
    common.state = lambda **kw: save(work / 'STATE.json', dict(utc=datetime.now(timezone.utc).isoformat(), **kw))
    common.night = types.SimpleNamespace(predictor=types.SimpleNamespace(load_routes=lambda: {p.name: None for p in sorted((common.DATA / 'task4').iterdir()) if p.is_dir()}))
    sys.modules['common'] = common
    for name in ['task4_prior_structure', 'task4_span_shrink_stress']:
        load(name, archive / (name + '.py'))
    source_protocol = out / 'task4_covariance_audit/protocol.json'
    source_protocol.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(here / 'results/task4_covariance_audit/protocol.json', source_protocol)
    covariance = load('portable_covariance_preparation', archive / 'task4_covariance_audit.py')
    for panel in common.night.predictor.load_routes():
        for split in ['train', 'validation', 'private']:
            covariance.record(panel, split)
    module = load('registered_background_mean', archive / 'task4_background_mean_audit.py')
    module.O = out / 'audit'
    module.O.mkdir(parents=True, exist_ok=True)
    module.main()
    save(out / 'portable_entrypoint.json', dict(entrypoint_sha256=sha(__file__), scope='Observable-space model check only; no hidden OD validation or new prediction.', official_score=None))


if __name__ == '__main__':
    main()
