# Task4 matching the complete span histogram

At a 1% total OD movement cap, require both near-optimal OD solutions to
match in every span bin. Their common histogram is free to differ from the
frozen baseline. Minimize the common multiplier on their maximum-cell-move
optima. The required slack is 10.77% on D12 I405 N and 7.87% on D7 I10 E.
The isolated run reproduces the thresholds. Different optimal solutions can
produce different common histograms. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_shared_histogram.py --release-root /path/to/kaggle_public --output /path/to/audit-output
