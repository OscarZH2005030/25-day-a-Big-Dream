# Weaker shared-span covariance

At a shared span variance of 3.125%, the balanced other-cell objective beat the total-cell objective on two to six of ten roads under the five joint covariance laws. The average difference ranged from −0.0729 to +0.0093 percentage points of initial weighted total variance. At 12.5%, the corresponding win count was ten of ten for all five laws.

The follow-up fixed its span weight before running. The origin and destination weights were each 10.9375%, preserving the per-cell variances and total shared variance. The paired differences use the common unmeasured set within each experiment. This set can change across covariance laws because the selected probes can change. The experiment conditions on hypothetical exact OD measurements and evaluates variance reduction.

From the experiment root, use the same environment and release as the parent comparison.

```sh
python task4/audit_direct_od_marginal_measurement_gain.py --release-root /path/to/release --output /path/to/weak --span-correlation --span-shared-weight 0.03125
python task4/span_covariance/compare_task4_span_covariance.py /path/to/original/task4_direct_od_marginal_measurement_gain /path/to/weak/task4_direct_od_custom_span_covariance /path/to/paired
```

The portable entry reproduced the input hashes and empty failure list. Its maximum absolute discrepancy was 4.00×10⁻¹⁵ in road-level results and 5.00×10⁻¹⁶ in the summary. Simulated S_OD, case scores, public count NMAE and weak-prior holdout error were unmeasured.
