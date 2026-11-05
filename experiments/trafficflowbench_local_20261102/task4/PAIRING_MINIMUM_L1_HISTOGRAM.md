# Pairing ambiguity near minimum prior distance with exact span counts

Fix baseline count predictions and origin and destination totals on ten roads and two splits. Separately with and without the full baseline span histogram, minimize the unrestricted-sign L1 distance to the released weak prior and bound the total flow carried by the half of OD cells with largest prior values. The endpoints allow numerical slack of 1e-8 times total demand.

With the histogram, mean interval widths are 4.06 percentage points in validation and 3.96 in private. Without it, they are 5.98 and 6.54. The regimes use separate L1 minima, so the intervals need not be nested. This is a conditional ambiguity audit using baseline quantities.

    python task4/audit_pairing_minimum_l1_histogram.py --release-root /path/to/release --output /path/to/output
