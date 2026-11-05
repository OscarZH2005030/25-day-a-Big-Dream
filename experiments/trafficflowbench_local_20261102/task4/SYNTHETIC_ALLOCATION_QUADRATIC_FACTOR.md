# Synthetic allocation recovery with an omitted quadratic span factor

Add an independent random quadratic-span factor to the same 92,160 seeded Gaussian draws. The candidate covariance grid omits this factor. Compare raw likelihood with two contrasts, first removing the prior and linear span directions and then also removing the quadratic direction.

At the largest amplitude, mean recovery is 93.36% raw, 96.63% after the two-direction contrast, and 96.55% after the three-direction contrast. The added factor and removed directions are prespecified. This tests a synthetic covariance misspecification and does not validate real count errors or hidden OD estimates.

    python task4/audit_synthetic_allocation_quadratic_factor.py --release-root /path/to/release --output /path/to/output
