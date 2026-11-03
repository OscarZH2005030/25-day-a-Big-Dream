# Cross-source agreement at matched movement

Seed 2026110349 covers eighty artificial cases across ten networks, two demand families and four error mechanisms. The reference source direction averages the two geometry-orthogonal source residuals. The agreement direction takes the smaller absolute residual where their signs agree and zero otherwise. Both apply 95th-percentile clipping and re-projection, so the final agreement direction can have nonzero entries outside the initial agreement set.

Both solutions use the same 5% local cap and 0.002 global cap, with realized movement matched case by case. The three non-OD score components stay fixed.

|Artificial mechanism|Mean-source mean OD change|Agreement mean OD change|OD harms mean to agreement / 20|
|---|---:|---:|---:|
|Independent lognormal|+0.000623|+0.000631|2 to 2|
|Independent positive background|+0.000635|+0.000758|1 to 2|
|Shared lognormal|+0.000033|+0.000057|10 to 10|
|Shared positive background|+0.000515|+0.000579|4 to 4|

The agreement rule improves all four means relative to averaging in this artificial draw. The differences are small, and the harm counts do not improve. Agreement can reflect shared error. These results do not identify hidden OD quality.

Independent readback verifies 240 actions, both local caps, matched global movement and score invariants. The portable run matches all four result files and eighty artifact hashes.

```powershell
python task4/stress_source_agreement.py --release-root <release-root> --output <new-output-directory>
```
