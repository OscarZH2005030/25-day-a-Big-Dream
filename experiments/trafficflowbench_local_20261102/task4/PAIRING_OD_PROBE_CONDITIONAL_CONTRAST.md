# Signed OD contrast with 256 matched probes

On I10 East, greedily select 256 cells by the residual diagonal of the nullspace projector for counts, origin and destination margins, and trip-span bins. At each step condition the projector on previously selected cells. Use 12-decimal rounding and path order to break ties.

Within the two near-optimal prior views, require the 256 probe values and span histograms to match. Keep the 1% movement budget and 125% maximum-cell caps. Minimize and maximize the signed full-OD difference, with signs taken from the rounded demand-normalized joint nullspace component of the pooled train/validation prior. No random draws are used.

The local feasible interval is -0.7494% to +0.7164% of reference demand. Both signs are feasible under these observation constraints. The contrast measures alignment with the released prior structure. Its interval concerns this road and this fixed measurement design, and cannot establish hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_conditional_contrast.py --release-root /path/to/kaggle_public --output /path/to/output
```
