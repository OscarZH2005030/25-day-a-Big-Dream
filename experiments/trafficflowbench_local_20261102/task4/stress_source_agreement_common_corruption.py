"""Reproduce the paired artificial shared-source corruption audit."""
from pathlib import Path
import runpy

import stress_source_agreement
from stress_observable_gate import load


if __name__ == '__main__':
    stress_source_agreement.main()
    archive = Path(__file__).resolve().parents[1] / 'archive/daytime'
    for name in ('od_cycle_refinement', 'task4_source_clip_stress', 'task4_count_invariant_drift'):
        load(name, archive / (name + '.py'))
    load('audit_source_agreement_common_corruption', archive / 'audit_source_agreement_common_corruption.py').main()
    runpy.run_path(str(archive / 'verify_source_agreement_common_corruption.py'))
