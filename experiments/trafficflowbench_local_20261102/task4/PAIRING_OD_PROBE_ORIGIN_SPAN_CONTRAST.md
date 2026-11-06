# Origin-specific long-trip counts and signed OD contrast

On I10 East, keep the 256 conditionally selected OD cells and require an additional match in the total flow from each origin on paths with spans strictly above the median path span. There are 64 nonempty aggregate rows. These rows describe hypothetical observations. The experiment uses the same two feasible prior views, trip-span histogram matching, movement limits, and signed contrast as the matched-probes audit.

The signed contrast ranges from -0.6918% to +0.6906% of reference demand in the local run. The interval is 5.7% narrower than the matched-probes interval and still includes both signs. Only the contrast endpoints incorporate the origin aggregates. The witness and minimum probe separation columns describe the intermediate probes-only solve. This comparison measures ambiguity under one assumed measurement design and supplies no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_origin_span_contrast.py --release-root /path/to/kaggle_public --output /path/to/output
```
