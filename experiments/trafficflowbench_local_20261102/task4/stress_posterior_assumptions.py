"""Reproduce all 96 fixed synthetic posterior decision cases; no release data needed."""
import argparse
import shutil
import sys
import types
from pathlib import Path
from datetime import datetime, timezone
from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    work = out / 'work'
    work.mkdir(parents=True, exist_ok=True)
    archive = Path(__file__).resolve().parents[1] / 'archive/daytime'
    common = types.ModuleType('common')
    common.DATA = common.WORK = common.ROOT = work
    common.OUT = out
    common.sha, common.save = sha, save
    common.check_deadline = lambda: None
    common.state = lambda **kw: save(work / 'STATE.json', dict(utc=datetime.now(timezone.utc).isoformat(), **kw))
    sys.modules['common'] = common
    for name in ['task4_prior_structure', 'task4_span_shrink_stress', 'task4_gaussian_posterior_pilot']:
        shutil.copy2(archive / (name + '.py'), work / (name + '.py'))
        load(name, archive / (name + '.py'))
    module = load('registered_smallnet_stress', archive / 'task4_smallnet_posterior_stress.py')
    module.O = out / 'stress'
    module.O.mkdir(parents=True, exist_ok=True)
    module.main()
    save(out / 'portable_entrypoint.json', dict(entrypoint_sha256=sha(__file__), scope='Known-latent artificial assumption stress, not real OD validation; all failures retained.', official_score=None))


if __name__ == '__main__':
    main()
