# Local projection of the two-source pairing target

Project the fixed two-source displacement onto directions that keep both signed moves feasible near the baseline. Keep count predictions, exact origin/destination totals and original weak-prior L1 distance fixed. With the baseline span histogram also fixed, the projected direction retains 86.1% of the squared norm of the pairing displacement. The mean feasible symmetric movement along that single direction is 0.038% of baseline demand in normalized L1. This is a geometric property of the fixed proxy target and baseline.

    python task4/audit_pairing_reversible_target.py --release-root /path/to/release --output /path/to/output
