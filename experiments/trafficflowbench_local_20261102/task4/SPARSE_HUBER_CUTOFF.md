# Huber cutoff sensitivity on sparse demand

In the earlier sparse lognormal cases, the positive median was about 1e-6 and every squared-initializer residual exceeded the physical Huber cutoff. The new fixed experiment compares that cutoff against a floor at the training mean. It retains the features, target normalization, penalty, folds, eight IRLS steps and 5% blend. Seed 2026110333 covers twenty new artificial sparse cases at count-shape TV 0.14.

|Artificial mechanism|Fit|Mean Task4 difference|Composite harms / 10|
|---|---|---:|---:|
|Shared lognormal|Squared|+0.000198|5|
|Shared lognormal|Median cutoff|+0.002413|0|
|Shared lognormal|Mean floor|+0.000708|0|
|Shared positive background|Squared|-0.001179|8|
|Shared positive background|Median cutoff|-0.001620|9|
|Shared positive background|Mean floor|-0.001577|9|

The mean floor reduced the gain in shared lognormal cases. Both Huber cutoffs performed worse on average than squared fitting under shared positive background. The diagnostic confirms sensitivity to the cutoff, and the artificial experiments offer no basis for selecting a real OD forecast.

Independent readback checked 80 action scores, the physical cutoffs and 200 objective pairs. Both tables and all 20 artifact hashes matched the portable rerun. The earlier 200-fold cutoff audit used exposed cases and remains descriptive.

```powershell
python task4/stress_sparse_cutoff.py --release-root <release-root> --output <new-output-directory>
```
