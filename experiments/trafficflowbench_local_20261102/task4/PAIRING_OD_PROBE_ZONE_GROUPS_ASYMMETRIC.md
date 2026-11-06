# Different tolerances for direct and grouped OD rows

On I10 East, keep 256 conditional direct OD measurements and 528 grouped totals. Allow up to 0.001% of reference demand (about 0.967 flow units) of pairwise disagreement on direct rows and twice that amount on grouped rows. Keep the same feasible sets and signed weak-prior contrast.

The local interval is -0.7172% to +0.7023% of reference demand. Its width is about 4.6% smaller than for 784 direct cells at the original tolerance, compared with 12.7% smaller when all rows use the original tolerance. Both signs remain feasible. Tolerances are hypothetical deterministic bounds. The witness columns refer to the intermediate probes-only solve. This single-road experiment provides no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_zone_groups_asymmetric.py --release-root /path/to/kaggle_public --output /path/to/output
```
