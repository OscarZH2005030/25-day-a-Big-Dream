"""Score fixed OD blends from an isolated stress_od_cycles.py run."""
import argparse
import sys
import types
from pathlib import Path

from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    common = types.ModuleType('common')
    common.WORK = args.cycle_run / 'work'
    common.OUT = args.output
    common.sha, common.save = sha, save
    sys.modules['common'] = common
    source = Path(__file__).resolve().parents[1] / 'archive/daytime/audit_task4_cycle_blend_probability.py'
    load('cycle_blend_archive', source).main(args.cycle_run / 'stress/all_cases.csv')


if __name__ == '__main__':
    main()
