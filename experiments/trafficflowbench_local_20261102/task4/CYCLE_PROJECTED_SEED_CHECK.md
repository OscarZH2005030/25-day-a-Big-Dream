# Rectangle seed sensitivity

Repeat feasible OD reprojection on the same 480 exposed synthetic cases with rectangle seed XOR 2026110642. Low-strength forward mean gain is +0.001315 versus +0.001329 with the previous rectangle seed. All 120 group means are positive in both runs. The second seed changes the projection search, while the synthetic truth cases stay fixed.

Generate inputs with task4/stress_od_cycles.py, then run:

```sh
python task4/audit_task4_cycle_projected_seed_check.py --cycle-run /path/to/cycle-run --release-root /path/to/kaggle_public --output /path/to/output
```
