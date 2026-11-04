# OD measurement design under interpolated variance laws

The four fixed selection rules use the original variance exponents 0, 0.5 and 1. This experiment also scores their selections at exponents 0.25 and 0.75, which are excluded from selection. All 120 original road-design-law rows match the prior experiment within 1e-12 absolute tolerance.

Across the ten roads, the linear-variance rule has a mean worst gap of 0.147 percentage points on the two interpolated laws. The square-root and balanced designs have gaps of 0.301 and 0.445 percentage points. Each gap uses the best tested design for that road and law as the reference. The covariances and exact direct OD measurements are hypothetical.

    python task4/audit_direct_od_variance_interpolation.py --release-root /path/to/release --output /path/to/output
