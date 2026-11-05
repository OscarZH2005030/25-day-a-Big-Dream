# OD pairing ambiguity after an exact span histogram

Across ten roads and both validation and private splits, bound the flow share carried by the half of OD pairs with highest weak-prior values. Preserve baseline count predictions, exact margins, and the exact original weak-prior L1 distance within the baseline orthant. Compare the ranges with and without the full baseline span histogram fixed.

All 20 paired intervals are nested. Average width falls from 13.19 to 9.80 percentage points in validation and from 13.26 to 9.95 in private. These feasible intervals are conditional on baseline quantities and the fixed sign orthant.

    python task4/audit_pairing_nonnegative_histogram_ranges.py --release-root /path/to/release --output /path/to/output
