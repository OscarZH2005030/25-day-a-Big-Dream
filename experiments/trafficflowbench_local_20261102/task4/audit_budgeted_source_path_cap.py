"""Rebuild the artificial source directions and reproduce the per-path cap."""
from pathlib import Path

import stress_budgeted_source_direction
from stress_observable_gate import load


if __name__ == '__main__':
    stress_budgeted_source_direction.main()
    archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    load('task4_budgeted_source_direction',archive/'task4_budgeted_source_direction.py')
    load('verify_cycle_budget',archive/'verify_cycle_budget.py')
    load('registered_path_cap',archive/'audit_budgeted_source_path_cap.py').main()
