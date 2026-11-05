# Task4 long-trip share across all ten roads

Extend the two-road audit to the remaining eight under the same constraints.
The interval of flow share above the median admissible OD span overlaps
between the two target rules on six of these eight roads. The fixed-component
minimum exceeds the half-average maximum on D7 I10 W and D7 I210 W.
Together with the pilot, the intervals overlap on eight of ten roads.
The result is specific to the median threshold and the feasible sets.
Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_long_trip_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
