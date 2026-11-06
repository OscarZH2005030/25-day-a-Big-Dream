# Can weak-prior fit predict better OD pairing?

On 960 previously exposed synthetic cases, predict whether 5% pair calibration beats 5% geometry calibration in synthetic S_OD. The sole predictor is cross-fitted weak-prior MSE reduction. Evaluate separately with each road or error family held out, comparing Brier loss against the corresponding training prevalence.

Mean Brier loss is 0.2472 versus 0.2498 with roads held out, and 0.3631 versus 0.3596 with error families held out. No case crosses the registered 0.65 probability threshold. The pair candidate wins 462/960 cases, including 203/240 under independent lognormal error and 10/240 under shared lognormal error. These frequencies describe the synthetic stress set. They do not estimate hidden OD accuracy.

```sh
python task4/audit_task4_pair_probability.py results/task4_crossmonth_shared_noise_stress_qr/all_cases.csv /path/to/output
```
