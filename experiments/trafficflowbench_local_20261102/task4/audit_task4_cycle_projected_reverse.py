"""Replay projected forward and reverse OD directions."""
import argparse
import sys
import types
from pathlib import Path

from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle-run', type=Path, required=True)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    common = types.ModuleType('common')
    common.WORK = args.cycle_run.resolve() / 'work'
    common.ROOT = common.WORK
    common.DATA = args.release_root.resolve()
    common.OUT = args.output
    common.sha, common.save = sha, save
    sys.modules['common'] = common
    archive = Path(__file__).resolve().parents[1] / 'archive/daytime'
    for name in ('task4_prior_structure', 'task4_span_shrink_stress', 'od_cycle_refinement'):
        load(name, archive / (name + '.py'))
    load('projected_reverse', archive / 'audit_task4_cycle_projected_reverse.py').main(
        args.cycle_run / 'stress/all_cases.csv')


if __name__ == '__main__':
    main()
