# Road-specific risk of OD flow blending

Split the existing 480 exposed synthetic cases by road, error family and drift. Keep the four blend weights and both endpoints unchanged. At weight 0.10 for cycle005, 6 of 120 joint strata have negative mean gains. The lowest mean is -0.0000440 on I405 North with shared positive background and replaced pairs. Each stratum has four cases. These are descriptive results on the existing scenarios, without hidden score validation.

Run task4/stress_od_cycles.py first, then run:

```sh
python task4/audit_task4_cycle_blend_joint.py --cycle-run /path/to/cycle-run --output /path/to/joint-run
```
