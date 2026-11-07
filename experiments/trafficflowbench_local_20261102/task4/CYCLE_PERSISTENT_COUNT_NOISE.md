# Persistent count error with matched marginal variance

Put the moderate count error variance entirely into a detector bias shared across months. On the same 480 synthetic cases, low-strength forward reprojection gives mean OD gain +0.000407171, 119/120 positive stratum means and worst individual gain -0.00190733. Mixed persistent/independent noise at the same marginal variance gave +0.000405172. Simulated count-fit NMAE is 0.975575%. Public count and weak-prior holdout metrics were not remeasured in this simulation. An interrupted first run was preserved and the identical retry was checked against a complete isolated regeneration.

```sh
python task4/stress_od_cycles_persistent_count_noise.py --release-root /path/to/kaggle_public --output /path/to/output
```
