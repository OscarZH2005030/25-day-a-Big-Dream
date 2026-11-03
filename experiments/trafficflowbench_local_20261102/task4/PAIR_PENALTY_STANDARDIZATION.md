# Fold-local pair-column standardization

The source pair residual had RMS from 1.69e7 to 1.31e8 in the prior sparse lognormal stress, versus 2.46 to 6.40 on the ten public source pairs. With a unit penalty on the original coefficient, the equivalent penalty on a unit-RMS column is the inverse squared RMS. This motivated a direct ablation of the pair coefficient penalty.

Seed 2026110334 covers 40 new artificial cases on ten networks, two demand families and two shared error mechanisms at count-shape TV 0.14. The standardized method divides only the pair column by its target training-fold RMS. Source construction, geometry, targets, folds, intercept treatment and 5% blend stay fixed. In the raw design this gives a penalty of RMS squared on the pair coefficient.

|Artificial mechanism|Original mean Task4 difference|Standardized mean difference|Composite harms for each / 20|
|---|---:|---:|---:|
|Shared lognormal|+0.001770|+0.001773|0|
|Shared positive background|-0.000516|-0.000519|16|

Standardizing the penalty did not materially change average effects or resolve the positive-background failure in this artificial comparison. Public activation and hidden OD accuracy remain unknown.

Independent readback checked 120 action scores and 200 fold scales and stationarity conditions. The full portable rerun matched both tables and all 40 case artifact hashes. The earlier penalty audit is descriptive and uses exposed cases.

```powershell
python task4/stress_pair_standardization.py --release-root <release-root> --output <new-output-directory>
```
