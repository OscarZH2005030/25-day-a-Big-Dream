# Identifiability of OD span thresholds

For each validation road, project every nonconstant integer threshold query for the number of trips exceeding a zone span onto the count nullspace and the count-plus-margin nullspace. All 795 threshold queries have a nonzero residual projection after adding hypothetical exact margins. The squared residual projection accounts for 1.81% to 97.92% of the query's squared norm. In contrast, the linear sum of zone spans is determined by the margins.

The projections test linear identifiability. They do not provide nonnegative feasible ranges at a particular observed margin or estimates of true trip distances.

    python task4/audit_pairing_span_threshold_geometry.py --release-root /path/to/release --output /path/to/output
