# Task4 finer amplitude control

Repeat the fixed half-average and 0.5 disagreement comparison within 20
equal-frequency source-amplitude bins per road and observation view. Paired
deltas subtract the lower-disagreement group from the higher-disagreement
group. All 400 paired strata contain both groups.

The higher-disagreement group has larger mean absolute and squared error
together in 166/200 strata under counts and margins and 156/200 with the
span histogram. It gains less from the half-average relative to zero on both
metrics in 155/200 and 167/200 strata, respectively. The paired output also
records the residual mean amplitude difference. These are descriptive
results with exposed weak-prior targets and shared road-level inputs.

From the experiment directory:

    python task4/audit_pairing_disagreement_fine_amplitude.py --release-root /path/to/kaggle_public --output /path/to/audit-output
