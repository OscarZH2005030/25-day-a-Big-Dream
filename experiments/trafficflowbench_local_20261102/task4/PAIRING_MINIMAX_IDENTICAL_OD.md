# Task4 threshold for identical OD flows

At the 1% movement cap, require both target rules to return the same
complete OD flow while retaining their own near-optimal primary L1
objectives. Both pilot roads are feasible. The minimum shared slack is
57.05% on D12 I405 N and 45.44% on D7 I10 E. The thresholds reproduce
in the isolated run. The common flow is not necessarily unique, so
its span summaries can vary. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_identical_od.py --release-root /path/to/kaggle_public --output /path/to/audit-output
