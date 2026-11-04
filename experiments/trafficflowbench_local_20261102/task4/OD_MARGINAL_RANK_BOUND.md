# Information bound for OD origin totals

After conditioning on exact released counts, any sixteen exact linear measurements reduce the whitened uncertainty in at most sixteen directions. The sum of the sixteen largest eigenvalues of the conditional origin-total covariance bounds the reduction in its trace. Measuring the corresponding principal components attains this bound in the linear covariance model.

Across ten roads, the bound averages 29.4% to 33.1% under three assumed cell variance laws. The minimum number of arbitrary exact linear measurements needed for half of the trace ranges from 13 to 42 across roads and laws. The bound allows arbitrary linear measurements and uses hypothetical covariance.

    python task4/audit_od_marginal_rank_bound.py --release-root /path/to/release --output /path/to/output
