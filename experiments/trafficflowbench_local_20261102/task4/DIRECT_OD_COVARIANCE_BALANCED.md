# OD selection balanced across two covariance models

Selection maximizes the worst normalized marginal gain over diagonal and joint covariance and three design exponents. All selections are evaluated under both covariances and five exponents. On the all-cell metric, the total objective ranges from 0.032 points below to 0.038 points above the better of the two single-covariance rules, averaged by road. The other-cell objective is below the comparator by 0.21 to 1.00 points. These are hypothetical measurement diagnostics.

    python task4/audit_direct_od_covariance_balanced.py --release-root /path/to/release --output /path/to/output
