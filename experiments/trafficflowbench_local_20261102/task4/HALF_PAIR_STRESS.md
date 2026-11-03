# Half-strength pair contribution

The earlier unit penalty changed the pair coefficient by at most about 0.21%. This registered comparison uses a penalty equal to the pair column's partial ridge information. That gives half the unpenalized pair coefficient while retaining the geometry penalty. Unit penalty and geometry-only fits are controls. All corrections use the same 5% prior blend, fold-local target and pair scales, and the same five folds.

Seed 2026110335 covers 40 new artificial cases with count-shape TV 0.14. The two error families and network inputs were studied previously.

|Artificial mechanism|Method|Mean Task4 difference|Composite harms / 20|
|---|---|---:|---:|
|Shared lognormal|Unit pair penalty|+0.001523|1|
|Shared lognormal|Half pair coefficient|+0.003336|2|
|Shared lognormal|Geometry only|+0.005248|2|
|Shared positive background|Unit pair penalty|-0.000255|14|
|Shared positive background|Half pair coefficient|-0.002801|16|
|Shared positive background|Geometry only|-0.005395|17|

Reducing the pair contribution improves the lognormal mean and worsens the positive-background mean in this stress. The geometry-only fit has the largest positive lognormal mean but also the worst individual lognormal result. These tradeoffs depend on the artificial mechanism. Hidden OD accuracy remains unknown.

Independent readback checked 160 action scores, 200 fold scales, the coefficient identity and the penalized first-order condition. The full portable run reproduced both tables and all 40 artifact hashes. The baseline and original data remain preserved.

```powershell
python task4/stress_half_pair.py --release-root <release-root> --output <new-output-directory>
```
