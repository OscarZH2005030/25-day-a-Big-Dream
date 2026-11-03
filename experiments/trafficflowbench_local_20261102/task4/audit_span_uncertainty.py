"""Reproduce span-dependent count uncertainty and optional 1200-case OD stress."""
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
    parser.add_argument('--with-stress', action='store_true')
    parser.add_argument('--with-od-directions', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    archive = Path(__file__).resolve().parents[1] / 'archive/daytime'
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
    covariance = load('portable_covariance_preparation', archive / 'task4_covariance_audit.py')
    for panel in common.night.predictor.load_routes():
        for split in ['train', 'validation', 'private']:
            covariance.record(panel, split)
    for name in ['task4_span_variance_audit', 'task4_variance_controls_audit'] + (['task4_weighted_variance_stress'] if args.with_stress else []):
        shutil.copy2(archive / (name + '.py'), work / (name + '.py'))
        module = load(name, archive / (name + '.py'))
        module.O = out / name
        module.main()
    if args.with_od_directions:
        for name in ['task4_gaussian_posterior_pilot', 'task4_smallnet_posterior_stress', 'task4_weighted_variance_stress']:
            shutil.copy2(archive / (name + '.py'), work / (name + '.py'))
            load(name, archive / (name + '.py'))
        module = load('registered_od_only_stress', archive / 'task4_od_only_direction_stress.py')
        module.O = out / 'task4_od_only_direction_stress'
        module.main()
    save(out / 'portable_entrypoint.json', dict(entrypoint_sha256=sha(__file__), scope='Public count likelihood and artificial stress only, not hidden OD validation or a submission.', official_score=None))


if __name__ == '__main__':
    main()
