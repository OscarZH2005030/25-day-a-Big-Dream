# Nonnegative ranges for long-trip share

Optimize the fraction of trips exceeding three prespecified weak-prior span quantiles. Preserve the frozen baseline's modeled counts, origin/destination totals and exact L1 distance to the original weak prior within its sign orthant. Solve 120 linear programs across ten roads and two splits. Report only aggregate endpoints and constraint drift.

At the median span threshold, the average feasible interval spans 19.6 percentage points in each split. These are feasible changes relative to the frozen baseline's constraints, not estimates of hidden demand. The fixed sign orthant restricts the feasible set further than the L1 equality alone.

    python task4/audit_pairing_nonnegative_span_ranges.py --release-root /path/to/release --output /path/to/output
