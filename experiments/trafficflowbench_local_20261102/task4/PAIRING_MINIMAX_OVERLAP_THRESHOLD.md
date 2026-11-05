# Task4 minimum shared slack for equal squared-span moments

At a 1% total movement cap, solve a joint LP for two feasible OD flows
with the same normalized squared-span moment. Each retains its own
near-optimal primary L1 objective. Minimize a common multiplier on
the two respective minimax optima, keeping the original additive
tolerances and the prior-distance bound.

The overlap threshold is 1.0536473 on D12 I405 N and 1.0469835 on D7 I10 E.
These are thresholds for the specified two optimization problems.
Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_overlap_threshold.py --release-root /path/to/kaggle_public --output /path/to/audit-output
