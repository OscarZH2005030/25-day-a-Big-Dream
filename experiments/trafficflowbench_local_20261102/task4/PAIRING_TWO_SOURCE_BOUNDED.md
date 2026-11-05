# Movement limits for two-source OD pairing projection

Limit L1 movement from baseline to 1%, 5%, or 10% of its total demand. Preserve baseline counts and margins, optionally preserve its span histogram, and cap original weak-prior distance at the baseline value. All 60 cases are feasible. Improvement in the signed-target proxy reaches the movement bound in 58 cases. Since L1 improvement cannot exceed L1 movement, this is a saturation result for the proxy. It gives no ranking against hidden OD truth. Different LP optima may have different prior distances.

    python task4/audit_pairing_two_source_bounded.py --release-root /path/to/release --output /path/to/output
