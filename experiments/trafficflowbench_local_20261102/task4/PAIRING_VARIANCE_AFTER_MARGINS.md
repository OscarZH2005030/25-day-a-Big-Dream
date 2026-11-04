# OD pairing variance after exact margins

Across ten private count geometries, use the pooled train and validation prior for weighting. Compare three variance exponents and three shared-origin and shared-destination correlations with matched cell variances. Condition on exact count rows and then on hypothetical exact complete origin and destination totals. Sequential and stacked projections give the same residual factors.

Across the nine combinations, the mean prior-weighted OD variance remaining after margins is 46.61% to 92.37% of the variance left after counts. These shares depend on the assumed error covariance and provide no hidden OD validation. Exact margins are hypothetical.

    python task4/audit_pairing_variance_after_margins.py --release-root /path/to/release --output /path/to/output
