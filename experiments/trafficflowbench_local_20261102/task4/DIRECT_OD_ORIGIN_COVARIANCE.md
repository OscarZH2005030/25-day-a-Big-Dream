# OD measurement design under shared origin error

The prior covariance assigns correlation 0.25 to errors sharing an origin and preserves the marginal variance of every cell. Both policies are scored on cells measured by neither policy. The other-cell policy has higher mean reduction at exponents 0 and 0.25, and lower reduction at 0.5, 0.75 and 1. This is a sensitivity check under a specified correlation model.

    python task4/audit_direct_od_origin_covariance.py --release-root /path/to/release --output /path/to/output
