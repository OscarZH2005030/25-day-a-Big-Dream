# Task4 matched-histogram OD gap at 25% slack across ten roads

Extend the 25% maximum-cell slack audit to the eight remaining roads.
The total OD movement cap is 1%, and each solution retains its own
near-optimal primary objective. The two solutions share a complete span
histogram. Across all ten roads the minimum OD L1 difference ranges from
0.175% to 0.329% of total demand. Histogram summaries can vary across
tied optimal solutions. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_histogram_intermediate_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
