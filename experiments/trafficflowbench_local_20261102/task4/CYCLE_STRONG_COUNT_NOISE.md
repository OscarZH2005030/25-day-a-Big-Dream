# Fixed OD reprojection under stronger synthetic count errors

Repeat the 480 synthetic cases at seed 2026110643 with detector-shared and independent count error log standard deviations of 0.06 and 0.04. This doubles both standard deviations relative to the preceding noise experiment.

Mean synthetic OD gain against the corresponding baseline is +0.000115123, compared with +0.000405172 under moderate noise and +0.001302683 with clean counts. Positive stratum means occur in 101 of 120 groups. The worst individual gain is -0.00106859. Simulated count-fit NMAE is 1.95151%. Public count and weak-prior holdout metrics were not remeasured. The results apply to the registered synthetic error model.

```sh
python task4/stress_od_cycles_strong_count_noise.py --release-root /path/to/kaggle_public --output /path/to/output
```
