# OD covariance aliases with matched cell variances

On each validation road, form two disjoint nullspace directions by splitting cells in stable order of train prior size and projecting the prior separately in each half. Their normalized sum and difference give unit-trace covariance factors with identical cell variances and identical covariance on counts and hypothetical exact margins.

The variance of the same train-prior linear functional differs in all ten constructed pairs. Ratios range from 35 to about 284,285. The upper value involves near cancellation in the denominator. These constructed rank-one examples quantify ambiguity and do not estimate real OD error.

    python task4/audit_pairing_matched_diagonal_aliases.py --release-root /path/to/release --output /path/to/output
