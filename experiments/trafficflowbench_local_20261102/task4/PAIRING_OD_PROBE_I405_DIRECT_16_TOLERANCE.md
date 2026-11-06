# I405 North OD contrast at common tolerance

Both hypothetical designs have 392 observation rows. Allow pairwise disagreement of 0.25 flow units per row. Retain exact base equalities and the same two near-optimal prior views. The replicated signed contrast intervals, as percentages of reference demand, are -0.0508 to +0.1122 for direct cells and -0.1223 to +0.1500 for the mixed design. The direct interval is about 40% narrower. Both include zero. This is a one-road sensitivity analysis with hypothetical observations and supplies no estimate of hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_i405_direct_16_tolerance.py --release-root /path/to/kaggle_public --output /path/to/direct
python task4/audit_pairing_od_probe_i405_mixed_16_tolerance.py --release-root /path/to/kaggle_public --output /path/to/mixed
```
