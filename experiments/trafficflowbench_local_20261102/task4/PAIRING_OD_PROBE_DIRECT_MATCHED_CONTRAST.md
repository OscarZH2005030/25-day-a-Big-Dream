# OD measurement designs with 784 hypothetical rows

Compare 784 direct OD cells chosen by conditional nullspace leverage with 256 cells chosen by the same rule plus 528 nonempty totals from the 32-by-32 contiguous partition. Use I10 East and the same signed weak-prior contrast and feasible sets in both cases.

The local interval for the direct-only design is -0.6513% to +0.6012% of reference demand. The mixed design gives -0.4972% to +0.4918%, an interval about 21.0% narrower. Both designs allow both signs. The nominal row counts are equal. Independence, acquisition cost and measurement noise may differ. These are hypothetical measurements on one road, and the bounds supply no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_direct_matched_contrast.py --release-root /path/to/kaggle_public --output /path/to/output
```
