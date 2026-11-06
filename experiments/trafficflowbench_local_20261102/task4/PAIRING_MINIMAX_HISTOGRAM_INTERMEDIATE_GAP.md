# Task4 matched-histogram OD gap at 25% slack

At a 1% total movement cap, increase the maximum cell movement cap
to 125% of each target rule's optimum. Retain the common complete span
histogram and the original primary objective tolerances. The minimum
OD L1 gaps on D12 I405 N and D7 I10 E are 0.315% and 0.175% of demand.
Both minima reproduce. Common histogram summaries can differ across
tied solutions. Hidden OD accuracy remains unverified.

    python task4/audit_pairing_minimax_histogram_intermediate_gap.py --release-root /path/to/kaggle_public --output /path/to/audit-output
