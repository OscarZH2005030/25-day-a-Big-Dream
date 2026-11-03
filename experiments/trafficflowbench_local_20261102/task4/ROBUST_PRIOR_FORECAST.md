# Huber target fitting under matched count shape

The fixed comparison uses the same pair features, five folds, ridge penalty and 5% prior blend for squared and Huber target fitting. The Huber threshold is one training positive median and the estimator takes eight IRLS steps from the squared solution. Seed 2026110332 covers ten networks, two demand families, two shared noise mechanisms, moderate drift and count-shape TV 0.14. All 40 registered cases completed.

|Artificial mechanism|Fit|Mean Task4 difference|Composite harms / 20|OD harms / 20|
|---|---|---:|---:|---:|
|Shared lognormal|Squared|+0.001211|2|4|
|Shared lognormal|Huber|+0.002775|1|3|
|Shared positive background|Squared|-0.000912|18|17|
|Shared positive background|Huber|-0.000650|16|16|

Huber improved on the squared fit in 17 of 20 composite comparisons in each mechanism. Against the original baseline, its mean remained negative in the positive-background mechanism, where the worst composite change was -0.003900. The fixed stress covers artificial errors on already examined network inputs. Hidden OD accuracy remains unknown.

Independent readback recomputed 120 action scores and 200 fold objective changes. The first absolute objective comparison failed on large-scale sparse fits. The maximum absolute discrepancy was 128, corresponding to at most 6.2e-16 relative to the initial objective across folds. Scores agreed within 1.2e-16. A complete portable rerun reproduced both tables and all 40 artifact hashes exactly. The comparison did not change the fits, scores or registered protocol.

```powershell
python task4/stress_robust_prior.py --release-root <release-root> --output <new-output-directory>
```
