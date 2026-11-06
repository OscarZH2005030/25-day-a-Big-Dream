"""Reproduce the Task4 refined contiguous origin-destination group contrast audit."""
import argparse
from pathlib import Path

import sys
import types

from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    common = types.ModuleType('common')
    common.ROOT = Path(__file__).resolve().parents[1]
    common.FROZEN_TASK4_BASELINE = Path(__file__).with_name('frozen_reference_path_flow.csv')
    common.DATA, common.OUT = args.release_root.resolve(), args.output.resolve()
    common.WORK = common.OUT / 'work'
    common.WORK.mkdir(parents=True, exist_ok=True)
    common.sha, common.save = sha, save
    common.state = lambda **info: save(common.WORK / 'STATE.json', info)
    common.night = types.SimpleNamespace(predictor=types.SimpleNamespace(
        load_routes=lambda: sorted(p.name for p in (common.DATA / 'task4').iterdir() if p.is_dir())))
    sys.modules['common'] = common
    archive = Path(__file__).resolve().parents[1] / 'archive/daytime'
    for name in ('task4_prior_structure', 'task4_span_shrink_stress', 'task4_crossmonth_shared_noise_stress',
                 'task4_crossmonth_shared_noise_qr', 'task4_exact_pair_audit', 'task4_extra_measurement_audit',
                 'audit_direct_od_measurement_design', 'audit_pairing_od_probe_zone_groups_refinement'):
        load(name, archive / (name + '.py'))
    sys.modules['audit_pairing_od_probe_zone_groups_refinement'].main()


if __name__ == '__main__':
    main()
