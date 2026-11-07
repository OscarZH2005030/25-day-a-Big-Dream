"""Reproduce the active Task4 prediction and public prior holdout audit."""
import argparse
import sys
import types
from pathlib import Path

from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    common = types.ModuleType('common')
    common.DATA = args.release_root.resolve()
    common.ROOT = common.WORK = args.output.resolve()
    common.OUT = args.output.resolve()
    common.sha, common.save = sha, save
    sys.modules['common'] = common
    archive = root / 'archive/daytime'
    for name in ('task4_prior_structure', 'task4_span_shrink_stress',
                 'task4_crossmonth_shared_noise_stress', 'task4_crossmonth_shared_noise_qr',
                 'task4_count_source_pair_stress', 'task4_exact_pair_audit',
                 'od_cycle_refinement', 'task4_od_cycle_stress'):
        load(name, archive / (name + '.py'))
    module = load('active_candidate_export', archive / 'export_task4_forward_public_audit.py')
    sys.argv = [str(__file__), str(root / 'task4/frozen_reference_path_flow.csv'),
                str(args.output / 'forward_candidate.csv')]
    module.main()


if __name__ == '__main__':
    main()
