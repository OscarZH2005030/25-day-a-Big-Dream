# Task4 adaptive direct OD probes

Starting with eight selected OD cells, add eight unselected cells with the
largest differences in the preceding minimizing witness pair. Test budgets
8, 16, 24 and 32 on two roads under the same broad feasible sets.

Both runs find pairs agreeing at all selected coordinates in every round.
At 32 probes, the full OD L1 difference exceeds 0.8% of demand on both roads.
The witness and the selected design can vary across runs. This experiment
uses no real OD measurements.

    python task4/audit_pairing_od_probe_adaptive.py --release-root /path/to/kaggle_public --output /path/to/audit-output
