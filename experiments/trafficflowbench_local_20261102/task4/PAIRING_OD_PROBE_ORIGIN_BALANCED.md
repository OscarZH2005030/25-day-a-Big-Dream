# Task4 origin-balanced direct OD probes

Rank OD entries by the mean normalized train and validation prior within
each origin. Select in origin-rank layers until reaching 256 on I10 East.
Both runs find a feasible pair with numerically identical readings and
a full OD L1 gap of approximately 0.718% to 0.719% of demand. These are
witness gaps, so differences from prior-ranked selection do not rank the
two designs by worst-case OD error. No actual OD measurements are used.

    python task4/audit_pairing_od_probe_origin_balanced.py --release-root /path/to/kaggle_public --output /path/to/audit-output
