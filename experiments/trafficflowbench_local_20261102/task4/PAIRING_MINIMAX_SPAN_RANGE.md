# Task4 span dispersion under two near-optimal objectives

On two prespecified roads, minimize and maximize the change in
flow-weighted squared zone span, normalized by the road's maximum squared
span and frozen demand. Hold the target L1 and maximum cell movement
objectives within 1e-8 of their optima relative to demand. Preserve frozen
counts and margins, nonnegativity, the original prior distance bound and
the 5% movement cap.

For D12 I405 N and D7 I10 E, the fixed-component range lies wholly above
the half-average range. The exact frozen-histogram control produces
changes of order 1e-18, which are numerical zero. These ranges describe
feasible flows under the specified optimization objectives on two roads.
Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_span_range.py --release-root /path/to/kaggle_public --output /path/to/audit-output
