# Source-guided directions at a matched L1 budget

Seed 2026110347 covers forty artificial cases at count-shape TV 0.14. Both LPs keep baseline counts, gross margins, original-prior orthant and prior L1 distance fixed. They maximize alignment with the exact source interaction or its 95th-percentile clipped and reprojected version. Both spend 0.002 of baseline total flow in L1 movement in every case.

|Artificial mechanism|Direction|Mean OD change|OD harms / 20|
|---|---|---:|---:|
|Shared lognormal|Raw source|-0.000597|12|
|Shared lognormal|Clipped source|-0.000526|10|
|Shared positive background|Raw source|+0.000115|10|
|Shared positive background|Clipped source|+0.000184|6|

Clipping improves both means relative to the raw direction, while both directions have negative lognormal means. The effects are specific to the tested LP objective and synthetic cases. Hidden OD accuracy remains unknown.

Independent readback checked 120 action scores, the matched budgets and the non-OD invariants. The portable run reproduced four result files and all forty artifacts exactly.

```powershell
python task4/stress_budgeted_source_direction.py --release-root <release-root> --output <new-output-directory>
```
