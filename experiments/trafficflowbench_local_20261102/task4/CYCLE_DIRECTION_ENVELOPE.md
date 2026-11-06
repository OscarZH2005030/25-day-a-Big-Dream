# Continuous ratio bound for two OD adjustment directions

On the 480 exposed synthetic cycle cases, bound the mean right derivative in each road/error/drift stratum by its supporting lines at five ratios. The derivative is concave in the ratio, so these lines give an upper bound at every ratio. A linear program maximizes the minimum of the lines over the entire unit interval.

The optimistic maximum worst-stratum derivative is -0.00043885. Thus every fixed convex mixture of the two directions has a stratum with negative mean derivative at zero movement. Concavity of the unclipped OD score along the movement segment implies a loss in that stratum for any positive movement weight. Both endpoint scores exceed the clipping floor. This numerical result concerns these directions and exposed synthetic cases only.

Generate inputs using task4/stress_od_cycles.py, then run:

```sh
python task4/audit_task4_cycle_direction_envelope.py --cycle-run /path/to/cycle-run --output /path/to/envelope-run
```
