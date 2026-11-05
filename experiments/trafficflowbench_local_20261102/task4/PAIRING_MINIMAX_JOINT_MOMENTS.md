# Task4 joint overlap of squared span and long-trip share

At 1% total movement, require the two solutions to have the same normalized
squared-span moment and the same flow share above the median admissible
OD span. Minimize their shared multiplier on the respective minimax optima
with the original constraints and objective tolerances.

The joint thresholds are 1.0542725 on D12 I405 N and 1.0474598 on D7 I10 E.
They exceed the squared-span-only thresholds by about 0.063 and 0.048
percentage points, respectively. These are feasibility results on two
roads. Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_joint_moments.py --release-root /path/to/kaggle_public --output /path/to/audit-output
