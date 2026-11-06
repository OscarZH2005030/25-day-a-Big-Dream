# Task4 OD differences with identical span histograms

At the minimum shared-histogram slack and 1% total movement, compare
the OD flows of the two solver-returned witnesses. Their normalized
L1 gaps are 0.887% of demand on D12 I405 N and 0.804% to 0.809% across
replicas on D7 I10 E. These are feasible examples. The gap can vary with
the chosen optimal witness. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_histogram_pair_gap.py --release-root /path/to/kaggle_public --output /path/to/audit-output
