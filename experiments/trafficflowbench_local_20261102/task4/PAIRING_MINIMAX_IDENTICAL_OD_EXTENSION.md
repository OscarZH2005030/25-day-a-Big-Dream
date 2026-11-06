# Task4 identical OD thresholds across ten roads

Repeat the exact OD equality problem on the remaining eight roads at a
1% total movement cap. Every road is feasible. The shared slack required
across the ten roads ranges from 45.44% to 63.45%, versus 7.63% to 12.36%
for equal span histograms. The isolated rerun reproduces the thresholds.
The common flow can vary across tied optima. Hidden OD remains unverified.

    python task4/audit_pairing_minimax_identical_od_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
