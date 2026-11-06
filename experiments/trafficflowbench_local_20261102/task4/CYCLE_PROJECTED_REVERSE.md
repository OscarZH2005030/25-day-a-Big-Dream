# Reprojected forward and reverse OD targets

On 480 previously exposed simulated cases, approach the original cycle endpoint or its reflection about the baseline with the same feasible rectangle sequence. Preserve count predictions, both margins and original prior L1 deviation.

The low-strength forward reprojection gains +0.001329 mean synthetic S_OD against baseline, with positive means in all 120 four-case strata and a worst individual loss of 0.002570. The original low-strength cycle endpoint had a larger mean gain of +0.002318. The reverse reprojections lose 0.001209 and 0.005068 on average. A new rectangle seed is used for this reprojection. The result does not establish hidden accuracy or robustness to fresh truth scenarios.

Generate the cycle inputs using task4/stress_od_cycles.py, then run:

```sh
python task4/audit_task4_cycle_projected_reverse.py --cycle-run /path/to/cycle-run --release-root /path/to/kaggle_public --output /path/to/output
```
