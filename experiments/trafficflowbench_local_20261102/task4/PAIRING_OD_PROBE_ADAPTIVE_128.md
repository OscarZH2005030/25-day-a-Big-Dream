# Task4 adaptive direct OD probes through 128 cells

Extend the same eight-cell adaptive batches to 128 probes on two roads.
At 128, both reruns find positive minimum measured L1 separation on D12
I405 N, from 31.92 to 35.84 flow units. On D7 I10 E both reruns still
find indistinguishable pairs with full OD L1 differences of 0.640%
to 0.652% of demand. The selected cells can change with solver ties.
Results describe the tested feasible sets and use no actual OD measurements.

    python task4/audit_pairing_od_probe_adaptive_128.py --release-root /path/to/kaggle_public --output /path/to/audit-output
