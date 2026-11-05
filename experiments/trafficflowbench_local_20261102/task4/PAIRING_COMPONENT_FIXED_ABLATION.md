# Task4 fixed component ablation

Compare fixed scales for the histogram and remaining pairing components,
using train alone and the equal train-validation mean. On the pooled
predictor, coefficients 1.0 and 0.5 respectively give relative L1 of 0.61205
and squared error of 0.43615, compared with 0.61412 and 0.44582 with the
cross-corridor coefficients. Each component alone has higher errors.
These comparisons use exposed weak priors and leave hidden OD unverified.

From the experiment directory:

    python task4/audit_pairing_component_fixed_ablation.py --release-root /path/to/kaggle_public --output /path/to/audit-output
