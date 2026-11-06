# Task4 minimum OD gap with equal span histograms

At 1% total OD movement per solution, minimize pairwise OD L1 under
identical span histograms and a shared multiplier within 1e-8 of its
minimum. The original count, margin, prior-distance and target tolerances
are retained. The isolated rerun confirms minima of 0.881% and 0.706% of
total demand on D12 I405 N and D7 I10 E, respectively. These minima concern
the specified feasible sets. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_histogram_min_gap.py --release-root /path/to/kaggle_public --output /path/to/audit-output
