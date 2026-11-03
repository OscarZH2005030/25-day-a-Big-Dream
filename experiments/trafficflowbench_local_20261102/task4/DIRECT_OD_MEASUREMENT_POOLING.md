# Pooling historical weights for hypothetical direct OD measurements

The design selects sixteen OD cells per road using train weights, validation weights, or a fixed equal mixture of both normalized weak priors. Private weights evaluate all three choices on identical private count and hypothetical gross-margin geometry. The same-split private selection is a comparator.

Across ten roads, greedy selection reduces private-weighted trace by 12.87% with pooled weights, 12.62% with train weights, 12.01% with validation weights, and 17.59% with private weights. These are changes in an assumed covariance, conditional on hypothetical direct measurements. No actual OD observation or hidden score is estimated. The pooled average is descriptive across ten roads.

    python task4/audit_direct_od_measurement_pooling.py --release-root /path/to/release --output /path/to/output
