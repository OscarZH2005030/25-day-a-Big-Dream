# Grouped OD tolerance crossover on I10 East

Keep the tolerance for direct OD cells at 0.001% of reference demand (about 0.967 flow units). Compare the 784 direct cells with the design using 256 direct cells and 528 grouped totals. At three times the direct tolerance on grouped rows, the mixed interval is -0.7478% to +0.7259% of reference demand and is 0.93% narrower than the direct interval. At four times, it is -0.7613% to +0.7367% and is 0.71% wider. The mixed feasible sets expand monotonically as grouped tolerance increases. Both signs remain feasible throughout. This is a bracket for this hypothetical design on one road. It gives no empirical measurement error rate or hidden OD accuracy estimate.

```sh
python task4/audit_pairing_od_probe_zone_groups_triple.py --release-root /path/to/kaggle_public --output /path/to/output3
python task4/audit_pairing_od_probe_zone_groups_fourfold.py --release-root /path/to/kaggle_public --output /path/to/output4
```
