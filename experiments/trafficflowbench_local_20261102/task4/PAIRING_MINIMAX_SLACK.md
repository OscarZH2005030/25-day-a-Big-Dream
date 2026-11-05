# Task4 sensitivity to the maximum cell movement bound

At a 1% total L1 movement cap, allow the largest single-cell movement
to be up to 125% of its minimax optimum. Keep the primary target tolerance
and the original prior-distance bound. On both pilot roads, the two target
rules now have overlapping normalized squared-span intervals. Under the
tighter minimax bound, both intervals were disjoint. This result concerns
two roads and leaves hidden OD accuracy unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_slack.py --release-root /path/to/kaggle_public --output /path/to/audit-output
