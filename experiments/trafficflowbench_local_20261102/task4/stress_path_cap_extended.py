"""Rebuild the artificial source directions and run fresh four-mechanism path-cap cases."""
from pathlib import Path
import shutil

import stress_budgeted_source_direction
from stress_observable_gate import load


if __name__ == '__main__':
    stress_budgeted_source_direction.main()
    import common
    archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    load('task4_budgeted_source_direction',archive/'task4_budgeted_source_direction.py')
    load('verify_cycle_budget',archive/'verify_cycle_budget.py')
    shutil.copy2(archive/'audit_budgeted_source_path_cap.py', common.WORK/'audit_budgeted_source_path_cap.py')
    load('audit_budgeted_source_path_cap',archive/'audit_budgeted_source_path_cap.py')
    load('registered_path_cap_extended',archive/'task4_path_cap_extended.py').main()
