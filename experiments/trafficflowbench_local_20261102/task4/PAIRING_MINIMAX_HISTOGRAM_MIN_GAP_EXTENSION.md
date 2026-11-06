# Task4 minimum OD gap across ten roads

Extend the minimum pairwise L1 problem to eight additional roads. Both
solutions share the complete span histogram, satisfy their own near-optimal
primary target, and remain within 1e-8 of the minimum shared multiplier.
With a 1% movement cap per solution, the minimum gap across all ten roads
ranges from 0.706% to 0.968% of demand. The two runs reproduce the optimum
within the numerical tolerance. Common histogram values can vary with the
optimal witness. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_histogram_min_gap_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
