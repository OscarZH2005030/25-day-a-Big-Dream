# Task4 prior-distance bound sensitivity

Repeat the two-road, 1% movement audit while allowing original prior L1
distance to increase by 1% of frozen demand. The triangle inequality makes
this relaxed bound redundant under the movement cap. Both roads retain
disjoint normalized squared-span ranges. The gaps between the fixed
component minimum and half-average maximum are approximately 0.00009908
and 0.00005283. This is a feasibility result on two roads. Hidden OD
accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_prior_ablation.py --release-root /path/to/kaggle_public --output /path/to/audit-output
