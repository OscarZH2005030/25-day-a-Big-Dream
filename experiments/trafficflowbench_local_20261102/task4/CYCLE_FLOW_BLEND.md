# Deterministic OD flow blends

On 480 exposed synthetic cases, interpolate the fixed cycle005 and cycle025 endpoints with their baselines using candidate weights of 0.10, 0.25, 0.50 and 0.75. Compare synthetic S_OD with the expected score of randomly choosing between the same endpoints. In these cases the score is above its clipping floor, so convexity of L1 favors the deterministic blend over random selection in expected score. Compare against baseline separately.

For cycle005 at 0.10, the worst noise-and-drift group mean gain is approximately 0.00000075 versus an expected loss of 0.000053 for random choice. Ten cases in that group still lose score against baseline. Larger tested weights or the cycle025 endpoint can have negative group means. The exposed simulation supplies no estimate of hidden score.

```sh
python task4/stress_od_cycles.py --release-root /path/to/kaggle_public --output /path/to/cycle-run
python task4/audit_task4_cycle_blend_probability.py --cycle-run /path/to/cycle-run --output /path/to/blend-run
```
