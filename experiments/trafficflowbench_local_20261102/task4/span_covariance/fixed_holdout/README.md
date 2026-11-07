# Task4 measurement objectives on a fixed OD target set

This ten-road experiment reserves the same OD cells under all three span covariance assumptions. Sixteen hypothetical exact probes must come from the remaining cells. The comparison uses the previous train/validation prior mixture, private count geometry and five marginal variance laws. A deterministic hash with seed 2026110707 reserves approximately one quarter of the cells on each road. The protocol and mask hashes record the selection.

The balanced reserved-cell objective uses the reserved cells in its selection loss. The comparison measures the additional reduction in reserved-cell variance relative to the total-cell objective, as percentage points of the initial weighted reserved-cell variance. These are the ranges over the five joint evaluation laws.

| Shared span variance | Mean gain over ten roads | Roads with a positive gain |
| --- | --- | --- |
| 0% | +0.3109 to +3.3517 points | 10 of 10 |
| 3.125% | +0.1038 to +3.2444 points | 8 to 10 of 10 |
| 12.5% | −0.0176 to +2.3255 points | 5 to 10 of 10 |

The road table also records the other-cell objective and diagonal evaluation. The covariance and exact-probe assumptions define the variance diagnostic. Public priors and known network geometry were available throughout. Simulated S_OD, case scores, public count NMAE and weak-prior holdout error were unmeasured.

From the experiment root, run the release entry point once per span weight in the existing NumPy, Pandas and SciPy environment.

```sh
python task4/audit_direct_od_marginal_measurement_gain.py --release-root /path/to/release --output /path/to/replay/zero --span-correlation --span-shared-weight 0 --fixed-holdout
python task4/audit_direct_od_marginal_measurement_gain.py --release-root /path/to/release --output /path/to/replay/weak --span-correlation --span-shared-weight 0.03125 --fixed-holdout
python task4/audit_direct_od_marginal_measurement_gain.py --release-root /path/to/release --output /path/to/replay/strong --span-correlation --fixed-holdout
python task4/span_covariance/fixed_holdout/compare_task4_fixed_od_holdout.py /path/to/replay --portable-layout
```

The release entry reproduced all input and mask hashes, the empty failure records and road metrics within 1.27×10⁻¹⁴ absolute difference.
