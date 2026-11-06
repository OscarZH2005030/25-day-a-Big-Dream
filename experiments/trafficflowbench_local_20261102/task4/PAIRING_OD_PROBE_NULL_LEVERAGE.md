# OD probes selected by nullspace leverage

On I10 East, select 256 OD cells by descending diagonal of the orthogonal projector onto the joint nullspace of observed counts, origin and destination margins, and trip-span bins. Round to 12 decimal places and break ties by path order. No random draws. The same two near-optimal prior views and 1% movement budget are used as in the origin-balanced experiment.

Both runs find zero L1 separation on the selected cells. The resulting feasible witness pairs differ across the full OD vectors by 0.856% to 0.859% of reference demand in L1. Different optimal witnesses can yield different full-vector differences. These are examples of ambiguity for this fixed design, with no upper bound on that ambiguity and no estimate of hidden OD accuracy.

Run from the experiment directory with Python and the public release available:

```sh
python task4/audit_pairing_od_probe_null_leverage.py --release-root /path/to/kaggle_public --output /path/to/output
```
