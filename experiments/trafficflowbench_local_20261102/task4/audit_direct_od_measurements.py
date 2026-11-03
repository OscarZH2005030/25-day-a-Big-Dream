"""Reproduce the hypothetical direct OD measurement design."""
import argparse
from pathlib import Path
import runpy
import sys
import types

from stress_observable_gate import load, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    common = types.ModuleType('common')
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
                 'audit_direct_od_measurement_design'):
        load(name, archive / (name + '.py'))
    sys.modules['audit_direct_od_measurement_design'].main()
    runpy.run_path(str(archive / 'verify_direct_od_measurement_design.py'), run_name='__main__')


if __name__ == '__main__':
    main()
