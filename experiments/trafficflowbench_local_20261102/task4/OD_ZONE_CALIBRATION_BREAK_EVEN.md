# Break-even precision for fixed-budget sensor calibration

With a budget of 16 sensors, compare 16 biased origin-total measurements with 15 origin-total measurements and one hypothetical reference that measures the shared bias. The analytical threshold is the largest reference-error variance, relative to shared-bias variance, that breaks even. Zero denotes a case with no strictly profitable precision under the numerical tolerance. The last selected origin is the fixed deletion. The best deletion uses the evaluation covariance and serves as an oracle diagnostic.

Perfect calibration can help after the fixed deletion in 21 of 120 cases at the lower independent-noise level and 79 of 120 at the higher level. Among those cases, the median thresholds are 0.02912 and 0.02906 respectively. The oracle yields positive thresholds in 48 and 109 cases. The analysis assumes a reference that directly measures the common bias and does not validate hidden OD estimates.

    python task4/audit_od_zone_calibration_break_even.py --release-root /path/to/release --output /path/to/output
