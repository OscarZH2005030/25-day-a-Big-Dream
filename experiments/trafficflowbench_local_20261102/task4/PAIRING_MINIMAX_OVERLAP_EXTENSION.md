# Task4 minimum shared slack across ten roads

Extend the joint LP to the eight remaining roads at the 1% movement cap.
The smallest common multiplier permitting equal normalized squared-span
moments ranges from 1.01916 to 1.05783 across all ten roads, including
the two pilot roads. These thresholds describe the specified constraints
and leave hidden OD accuracy unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_overlap_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
