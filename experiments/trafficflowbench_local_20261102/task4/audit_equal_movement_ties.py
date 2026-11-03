"""Rebuild the tied optima and reproduce the equal-movement comparison."""
from pathlib import Path

import stress_invariant_invisible_shrink
from stress_observable_gate import load


if __name__ == '__main__':
    stress_invariant_invisible_shrink.main()
    archive=Path(__file__).resolve().parents[1]/'archive/daytime'
    load('registered_shrink_tiebreak',archive/'audit_invariant_shrink_tiebreak.py').main()
    load('verify_cycle_budget',archive/'verify_cycle_budget.py')
    load('registered_equal_movement',archive/'audit_equal_movement_ties.py').main()
