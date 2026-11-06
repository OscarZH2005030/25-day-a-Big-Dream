# Sensitivity to measurement tolerance in the mixed OD design

On I10 East, retain the 256 conditional direct OD cells and 528 grouped OD totals. Allow each of the 784 hypothetical measurement values to differ between the two feasible solutions by up to 0.001% of reference demand, about 0.967 flow units per row. Keep the original base equalities and the same signed contrast.

The contrast interval is -0.6507% to +0.6478% of reference demand, about 31.3% wider than under exact matching. The tolerance is a deterministic bound on pairwise disagreement. The witness columns describe the intermediate probes-only solve. These bounds quantify sensitivity under one hypothetical design and provide no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_zone_groups_tolerance.py --release-root /path/to/kaggle_public --output /path/to/output
```
