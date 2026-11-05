# Task4 retrospective scale diagnostic

Minimize each proxy loss over a nonnegative scalar on the whole predictor or
on one orthogonal component while holding the other component fixed. L1
uses a weighted median of ratios, and squared loss uses a scalar projection.
The three target rotations reuse exposed weak priors.

Across rotations, the median multiplier on the histogram-determining
component is 1.77 to 1.81 for L1 and 1.94 to 2.06 for squared error. For the
histogram-nullspace component it is 0.86 to 0.91 for L1 and 1.10 to 1.21 for
squared error. These retrospective optima depend on the targets and cannot
serve as validation or deployment coefficients.

From the experiment directory:

    python task4/audit_pairing_scale_oracle_diagnostic.py --release-root /path/to/kaggle_public --output /path/to/audit-output
