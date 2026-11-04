# OD measurement design under shared origin and destination errors

The covariance combines separate origin and destination factors, each contributing 0.125 to correlation, and retains every marginal variance. The comparison uses OD cells measured by neither policy. The other-cell policy gives higher mean reduction at variance exponents 0 and 0.25 and lower mean reduction at 0.5, 0.75 and 1. The correlations and exact measurements are hypothetical.

    python task4/audit_direct_od_dual_covariance.py --release-root /path/to/release --output /path/to/output
