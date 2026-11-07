# Task4 OD measurement sensitivity to shared trip span

With correlated errors for equal zone-index spans, the balanced other-cell objective reduced variance on the common unmeasured set more than the balanced total-cell objective on all ten roads under each of five marginal variance laws. The mean extra reduction ranged from 0.196 to 0.431 percentage points of the initial weighted total variance. With diagonal evaluation covariance, the other-cell objective lost on six to eight roads. In the earlier origin-and-destination covariance experiment, it won on two to five roads under the corresponding joint laws.

The deterministic comparison uses the private count geometry, train/validation prior mixture, 16 hypothetical direct OD measurements, and the same per-cell marginal variances in both experiments. Shared variance weights are 0.125 for each margin in the original experiment and 0.0625 for each margin plus 0.125 for exact span in the new one. Exact span is the destination zone index minus the origin zone index. The common unmeasured set excludes the union of six designs separately in each experiment. The paired differences in this directory therefore compare selection policies within each experiment. Their absolute common-set reductions refer to different cell sets across experiments.

The covariance laws are assumptions for a sensitivity analysis. The experiment estimates the value of hypothetical measurements under those assumptions. Simulated S_OD and case scores, public count NMAE, and weak-prior holdout errors were unmeasured. `failures.json` records solver failures. The input and replay hashes cover the registered release files and numerical outputs.

Run from the experiment root using the published release and the existing NumPy, Pandas and SciPy environment.

```sh
python task4/audit_direct_od_marginal_measurement_gain.py --release-root /path/to/release --output /path/to/original
python task4/audit_direct_od_marginal_measurement_gain.py --release-root /path/to/release --output /path/to/span --span-correlation
python task4/span_covariance/compare_task4_span_covariance.py /path/to/original/task4_direct_od_marginal_measurement_gain /path/to/span/task4_direct_od_span_covariance /path/to/paired
```
