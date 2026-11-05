# Long-trip ambiguity at minimum weak-prior distance

Preserve modeled counts and exact origin and destination totals of the frozen baseline. Minimize L1 distance to the released weak prior, then bound median-span long-trip share at the minimum plus numerical tolerance and at the midpoint to the baseline budget.

At minimum distance the mean feasible interval is 8.65 percentage points in validation and 9.15 in private. At the midpoint it is 16.67 and 16.72. These are feasible intervals under baseline-conditioned assumptions, without hidden OD validation.

    python task4/audit_pairing_minimum_prior_ranges.py --release-root /path/to/release --output /path/to/output
