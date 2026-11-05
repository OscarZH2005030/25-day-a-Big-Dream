# Task4 rotation of the weak-prior target

Keep the private count incidence and margins fixed. Rotate the target across
the three released weak priors and form the predictor from the other two.
The fixed whole-prediction multiplier of 1.03 lowers both mean proxy errors
in all three rotations. Scaling only the histogram-nullspace component
raises mean relative L1 in all three rotations.

Every target has been exposed. Earlier-target rotations can use a later
prior as a predictor, so they are descriptive checks and provide no blind
or forward validation. Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_scale_split_rotation.py --release-root /path/to/kaggle_public --output /path/to/audit-output
