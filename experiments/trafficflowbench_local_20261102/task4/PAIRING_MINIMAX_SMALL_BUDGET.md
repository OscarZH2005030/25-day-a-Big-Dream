# Task4 span dispersion with a 1% movement cap

Repeat the squared-span range audit on the two pilot roads, reducing the
total L1 movement cap to 1% of frozen demand. Both roads still have disjoint
intervals. The gap between the fixed-component minimum and half-average
maximum is 0.00010016 on D12 I405 N and 0.00005626 on D7 I10 E. These are
normalized span second moments, under the given constraints and objectives.
Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_small_budget.py --release-root /path/to/kaggle_public --output /path/to/audit-output
