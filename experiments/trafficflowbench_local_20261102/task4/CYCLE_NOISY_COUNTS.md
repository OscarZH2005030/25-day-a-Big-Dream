# Fixed OD reprojection with synthetic count errors

Repeat the 480 draws at seed 2026110643 with identical truth, priors and folds. Multiply counts by synthetic lognormal errors with detector-shared log SD 0.03 and independent log SD 0.02. The mean injected count NMAE is 2.862%. These errors are hypothetical.

For low-strength forward reprojection, synthetic OD gain against the matching baseline changes from +0.001303 to +0.000405, positive group means from 119/120 to 120/120, and worst individual gain from -0.002920 to -0.002389. Simulated count-fit NMAE against each input changes from 0.002034% to 0.974276%. Public count and weak-prior holdout metrics were not remeasured. The experiment provides no hidden accuracy estimate.

```sh
python task4/stress_od_cycles_noisy_counts.py --release-root /path/to/kaggle_public --output /path/to/output
```
