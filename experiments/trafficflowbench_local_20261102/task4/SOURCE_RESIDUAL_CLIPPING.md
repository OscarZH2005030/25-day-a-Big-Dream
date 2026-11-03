# Clipping cross-input source residuals

The source residual removes the full geometry basis. The fixed intervention clips its magnitude at the 95th percentile and projects it out of geometry again. Both raw and clipped columns use fold-local unit RMS, the same ridge fit, five folds and a 5% target-prior blend. The threshold uses only the two source inputs.

Seed 2026110337 covers 40 new artificial cases on ten networks, two demand families and two shared noise mechanisms, at count-shape TV 0.14. All cases completed.

|Mechanism|Fit|Mean Task4 difference|Mean OD difference|Composite harms / 20|
|---|---|---:|---:|---:|
|Shared lognormal|Raw|+0.001166|+0.002061|2|
|Shared lognormal|Clipped|+0.006022|+0.013183|0|
|Shared positive background|Raw|-0.000472|-0.001333|17|
|Shared positive background|Clipped|+0.000592|+0.000967|9|

Clipping improved both artificial means. Positive-background harm persists in nine cases, and the network inputs and mechanism families were previously studied. The experiment establishes a candidate for further stress on broader artificial assumptions. Hidden OD accuracy is unknown.

Independent readback checked 120 actions and 400 fits for source cutoff, geometry orthogonality, fold RMS and ridge stationarity. The first verifier invocation completed the portable run but looked in the wrong output directory because a loop variable shadowed the directory name. Correcting the lookup verified exact table and artifact agreement across all 40 cases. The experimental code and results stayed fixed.

```powershell
python task4/stress_source_clip.py --release-root <release-root> --output <new-output-directory>
```
