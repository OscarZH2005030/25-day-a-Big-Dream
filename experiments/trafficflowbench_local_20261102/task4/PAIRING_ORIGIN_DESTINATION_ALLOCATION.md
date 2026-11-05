# Allocating shared OD error between origins and destinations

On ten road networks, hold each cell's total and shared variance fixed while varying the origin share of shared variance from zero to one in quarter steps. Condition on exact counts and hypothetical exact margins. The balanced allocation reproduces the earlier cell-scaled experiment.

With variance proportional to pooled prior demand, the average span across the five allocations is 10.62% or 13.32% of the balanced design's residual pairing variance, depending on the overall shared fraction. The span is at numerical precision for uniform cell variances. These are covariance sensitivities rather than validation of hidden OD demand.

    python task4/audit_pairing_origin_destination_allocation.py --release-root /path/to/release --output /path/to/output
