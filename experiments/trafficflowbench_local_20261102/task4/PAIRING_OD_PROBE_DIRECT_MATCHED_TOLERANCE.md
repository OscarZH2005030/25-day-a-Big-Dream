# Matched deterministic tolerance for direct and mixed OD measurements

On I10 East, allow up to 0.001% of reference demand (about 0.967 flow units) of pairwise disagreement on each of the 784 conditionally selected direct OD cells. Keep the same feasible sets and signed weak-prior contrast as the exact-match run. Compare against the mixed design with 256 direct cells and 528 group totals at the same per-row tolerance.

The local direct-only contrast interval is -0.7600% to +0.7276% of reference demand. The mixed-design interval is -0.6507% to +0.6478%, about 12.7% narrower. The width advantage was about 21.0% with exact matching. Both intervals include both signs. The tolerance is an assumed bound on pairwise disagreement. Measurement costs and empirical error distributions are unavailable, and this one-road experiment supplies no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_direct_matched_tolerance.py --release-root /path/to/kaggle_public --output /path/to/output
```
