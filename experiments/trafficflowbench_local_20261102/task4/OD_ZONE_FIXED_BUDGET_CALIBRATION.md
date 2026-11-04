# Fixed-budget calibration for hypothetical origin-total sensors

Across ten roads, four fixed designs, three prior variance laws and two noise levels, compare 16 biased origin measurements against 15 origin measurements and one pure-bias reference. The prespecified rule replaces the last origin selected by each design. A diagnostic oracle uses the evaluation covariance to choose the most favorable deletion from each fixed set.

The prespecified swap lost gain in all 720 cases. The oracle swap won in 6 cases, all with the highest reference precision. At that precision the average change from the prespecified swap was −0.736 and −0.556 percentage points in the two noise settings. These results are conditional on the assumed bias, calibration and covariance models. They do not validate hidden OD estimates.

    python task4/audit_od_zone_fixed_budget_calibration.py --release-root /path/to/release --output /path/to/output
