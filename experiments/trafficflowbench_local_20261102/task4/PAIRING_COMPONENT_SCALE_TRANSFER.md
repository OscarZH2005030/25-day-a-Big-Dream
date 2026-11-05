# Task4 cross-corridor component scaling

Fit coefficients on train-to-validation weak-prior prediction, excluding the
evaluation road and its opposite direction. Compare one shared coefficient
with separate coefficients for the span-histogram and remaining pairing
components. All coefficients minimize mean relative squared error on the
calibration roads, subject to [0,1], then stay fixed for private evaluation.

Separate coefficients improve both proxy losses on all 10 evaluation roads
relative to shared scaling. The mean relative L1 changes from 0.68173 to
0.63599 and mean relative squared error from 0.53365 to 0.46793. The private
weak-prior targets are exposed, and the opposite directions share corridors.
Hidden OD accuracy remains unverified.

From the experiment directory:

    python task4/audit_pairing_component_scale_transfer.py --release-root /path/to/kaggle_public --output /path/to/audit-output
