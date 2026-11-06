# I405 North mixed OD measurement feasibility

On I405 North, the joint count, margin and span nullspace has dimension 780 at the registered rank tolerance, below the requested 784 independent conditional direct probes. The original selector stopped on rank exhaustion. The mixed design uses 256 conditional direct probes and 528 nonempty contiguous group totals. Exact equality is infeasible across the two specified near-optimal prior views. Both contrast endpoint failures are retained.

Minimizing the maximum absolute discrepancy across the 784 mixed rows gives 0.7234 flow units in both runs. The earlier witness columns describe the intermediate probes-only solve. This is a separation between the specified feasible sets under hypothetical measurements. It gives neither a real sensor error threshold nor hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_i405_feasibility.py --release-root /path/to/kaggle_public --output /path/to/output
```
