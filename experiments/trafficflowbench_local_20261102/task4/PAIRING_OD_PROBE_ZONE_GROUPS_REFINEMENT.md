# Refined zone groups and signed OD contrast

On I10 East, keep the 256 conditionally selected OD cells and refine the contiguous 8-by-8 partition to 32 groups on each axis. Sort unique zones separately for origins and destinations and use floor(32 * zero-based rank / zone count). Require equality between the two feasible OD solutions on each of the 528 nonempty group totals. These are hypothetical measurements.

The local interval for the same signed weak-prior contrast is -0.4972% to +0.4918% of reference demand. Its width is 32.5% smaller than with 256 cells alone, and both signs remain feasible. Only the contrast endpoints incorporate the group totals. The witness and minimum separation columns come from the intermediate probes-only solve. The result concerns this measurement design on one road and supplies no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_zone_groups_refinement.py --release-root /path/to/kaggle_public --output /path/to/output
```
