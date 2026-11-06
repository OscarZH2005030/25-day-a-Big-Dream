# Continuous OD blend direction

On 480 exposed synthetic cases, compute the right derivative of the OD score at zero weight for both fixed cycle endpoints. Both endpoints have positive OD scores in all cases, so the unclipped L1 score is concave throughout each segment. A negative mean right derivative implies a lower mean score than baseline at every positive common blend weight in that stratum.

There are 3 of 120 negative road/error/drift strata for cycle005 and 10 of 120 for cycle025. This rules out curing those synthetic strata by shrinking the same direction. It gives no conclusion about hidden accuracy or other directions.

Generate the inputs with task4/stress_od_cycles.py, then run:

```sh
python task4/audit_task4_cycle_blend_direction.py --cycle-run /path/to/cycle-run --output /path/to/direction-run
```
