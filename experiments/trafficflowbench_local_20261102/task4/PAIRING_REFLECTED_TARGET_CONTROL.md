# Reflected target control for OD pairing

Reflect the two-source proxy target through the frozen baseline and compare both directions at the same 5% normalized L1 movement budget. Preserve counts and margins, optionally preserve the full span histogram, and cap weak-prior distance at the baseline distance.

Across ten roads and both views, optimizing either target reaches a gain of 0.05 and increases distance to the opposite target by 0.05. This makes the target-specific gain unsuitable by itself for ranking OD directions. Auxiliary metrics may vary among equally optimal flows.

    python task4/audit_pairing_reflected_target_control.py --release-root /path/to/release --output /path/to/output
