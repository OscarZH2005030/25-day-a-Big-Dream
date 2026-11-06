# OD pairing allocation across synthetic error families

Evaluate the expected synthetic S_OD of randomized choices among five fixed methods on 960 previously exposed scenarios. Optimize the worst mean gain over error-family and drift strata, and repeat with each error family held out. The allocation is an analysis of expected scores, without blending flow predictions.

The frozen baseline is the sole allocation with nonnegative mean gain in every tested stratum. Allowing a worst stratum mean loss of 0.001 or 0.005 permits up to 10.39% or 51.97% nonbaseline allocation, respectively. These are scenario-dependent loss budgets. They provide no real-world safety guarantee or hidden score estimate.

```sh
python task4/audit_task4_minimax_probability.py results/task4_crossmonth_shared_noise_stress_qr/all_cases.csv /path/to/output
```
