# Task4 adaptive direct OD probes through 256 cells

Extend the eight-cell adaptive batches on I10 East to 256 probes. This
road was selected because the 128-probe experiment retained ambiguity.
All measured gaps are numerically zero in both runs. At 256 probes,
the two witnesses differ by 0.463% to 0.494% of total OD demand across
replicas. The selected coordinates and the witnesses can vary. The
result concerns the tested designs and uses no actual OD observations.

    python task4/audit_pairing_od_probe_adaptive_256.py --release-root /path/to/kaggle_public --output /path/to/audit-output
