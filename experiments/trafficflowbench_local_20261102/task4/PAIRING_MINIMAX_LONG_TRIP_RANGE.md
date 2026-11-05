# Task4 long-trip share under near-optimal objectives

Bound the change in flow share above the unweighted median admissible OD
span on the two pilot roads. Keep nonnegativity, counts and margins, the
original prior-distance bound, and a 5% movement cap. Preserve the primary
and minimax objective values within 1e-8 of total demand.

The two target rules yield overlapping long-trip-share intervals on both
roads, unlike the disjoint squared-span intervals. Fixing the entire span
histogram makes the changes numerically zero. These results concern the
specified threshold and two roads. Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_long_trip_range.py --release-root /path/to/kaggle_public --output /path/to/audit-output
