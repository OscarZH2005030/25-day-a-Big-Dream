# Prior calibration at a public-range count shape mismatch

The max-across-input count shape TV on the ten public networks ranges from 0.106 to 0.190. Previous artificial cases reached at most 0.085. The new experiment applies a spatial tilt to each synthetic prior and targets TV 0.14 while preserving the observed total count. This matches one exposed input statistic. The generator still uses artificial truth and two specified noise families.

Seed 2026110331 covers ten networks, four demand families, moderate drift and two shared error mechanisms. All 80 registered cases reached the specified shape constraint. The same baseline solver and fixed 5% strength apply to geometry and cross-input pair forecasts.

|Artificial mechanism|Forecast|Mean Task4 difference|Composite harms / 40|OD harms / 40|
|---|---|---:|---:|---:|
|Shared lognormal|Geometry|+0.007144|0|1|
|Shared lognormal|Pair|+0.001869|2|2|
|Shared positive background|Geometry|-0.002600|31|35|
|Shared positive background|Pair|-0.000186|25|31|

The methods change rank across mechanisms at the same scalar mismatch. The bulk bias is zero up to numerical precision, so the prior count-total gate would admit these cases. Geometry's positive mean under shared lognormal does not transfer to shared positive background. The underlying mechanism and hidden OD accuracy on the public inputs remain unverified.

The portable runner reproduced the tables and all 80 artifact hashes. Independent readback verified 240 action scores and all count shape constraints. Generated inputs, per-path forecasts and coefficients remain local.

```powershell
python task4/stress_public_range_shape.py --release-root <release-root> --output <new-output-directory>
```
