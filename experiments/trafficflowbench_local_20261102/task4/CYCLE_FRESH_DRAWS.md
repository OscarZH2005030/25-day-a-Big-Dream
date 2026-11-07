# Fixed OD reprojection on fresh synthetic draws

Regenerate 480 synthetic cases under the existing scenario families with seed 2026110643. Keep the reprojection and rectangle seed rule fixed. Low-strength forward reprojection increases mean synthetic S_OD by 0.001303 against each case's baseline. Of 120 four-case strata, 119 improve on average. The lowest group mean is -0.00003315 on I405 South with shared positive background and moderate drift. The worst individual loss is 0.0029205. The new draws share the existing simulator assumptions and provide no estimate of hidden score.

```sh
python task4/stress_od_cycles_fresh.py --release-root /path/to/kaggle_public --output /path/to/output
```
