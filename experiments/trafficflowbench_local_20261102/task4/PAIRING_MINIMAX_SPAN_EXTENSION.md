# Task4 span dispersion across the remaining eight roads

Repeat the two-road audit under the same count and margin constraints and
optimization tolerances. On all eight additional roads, the minimum
normalized span second moment under the fixed-component target exceeds
the maximum under the half-average target. Together with the two pilot
roads, this holds on all ten audited roads. The intervals describe the
specified feasible sets and leave hidden OD accuracy unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_span_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
