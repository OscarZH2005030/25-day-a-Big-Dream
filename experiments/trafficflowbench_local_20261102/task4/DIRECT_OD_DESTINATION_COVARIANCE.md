# OD measurement design under shared destination error

The prior covariance assigns correlation 0.25 to errors sharing a destination while preserving every marginal variance. Both policies are scored on cells measured by neither policy. The other-cell policy gives higher mean reduction at exponents 0 and 0.25 and lower mean reduction at 0.5, 0.75 and 1. This sensitivity check uses hypothetical correlations and exact measurements.

    python task4/audit_direct_od_destination_covariance.py --release-root /path/to/release --output /path/to/output
