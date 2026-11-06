# Contiguous zone groups and signed OD contrast

On I10 East, keep the 256 conditionally selected OD cells. Sort unique origin and destination zones separately and assign each zero-based rank to group floor(8 * rank / zone count). Cross the origin and destination groups and require both solutions to match the total OD flow in each of the 36 nonempty group pairs. These aggregates are hypothetical. Use the same signed contrast, feasible prior views, and movement limits as in the matched-probes audit.

The local contrast interval is -0.6976% to +0.6799% of reference demand. Its width is 6.0% smaller than for the 256 probes alone. Both signs remain feasible. The 36 group totals and the 64 origin-specific long-trip totals are distinct designs with different numbers of aggregates. Only the contrast endpoints incorporate the group totals. The witness and minimum separation columns come from the intermediate probes-only solve. This single-road diagnostic supplies no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_zone_groups_contrast.py --release-root /path/to/kaggle_public --output /path/to/output
```
