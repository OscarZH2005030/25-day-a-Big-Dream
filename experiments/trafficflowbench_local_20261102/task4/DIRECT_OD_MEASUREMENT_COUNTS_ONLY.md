# Hypothetical direct OD selection with released counts only

The selection geometry uses only released count rows. It omits the hypothetical gross origin and destination totals from the earlier experiment. Sixteen measurements are selected using train, validation, pooled historical, or private weak-prior weights. The private weak prior evaluates all options.

On ten roads, greedy and largest-prior selection produce the same weighted trace ratios at the reported precision. Pooled historical weights reduce the private-weighted trace by 13.31%, compared with 12.95% for train weights and 11.73% for validation weights. Private weights give 17.93% as an in-split comparator. This is covariance reduction under hypothetical exact measurements, with no observed OD labels or hidden performance estimate.

    python task4/audit_direct_od_measurement_counts_only.py --release-root /path/to/release --output /path/to/output
