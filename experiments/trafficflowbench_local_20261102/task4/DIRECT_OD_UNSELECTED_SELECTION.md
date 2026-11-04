# Prioritizing uncertainty in unmeasured OD cells

This experiment removes a candidate cell's own variance gain from the greedy objective. It compares four selection rules on ten roads and five assumed variance laws, using sixteen hypothetical exact measurements per road. Selection uses pooled historical weights. Evaluation uses private weak-prior weights.

Under the square-root variance law, the balanced rule gains 0.236 percentage points in unmeasured-cell weighted trace reduction and loses 17.706 points in overall reduction relative to the original objective. The uniform-variance selection rule loses unmeasured-cell reduction when evaluated at exponents 0.5, 0.75 and 1. These comparisons use each policy's own unmeasured set, which can differ between policies. They describe hypothetical covariance reduction.

    python task4/audit_direct_od_unselected_selection.py --release-root /path/to/release --output /path/to/output
