# Task4 fixed-scale control

Use 1.03 as a single rounded representative of the exposed source-derived
factors. The multiplier was not selected against target errors. Compare it
against the unchanged half-average and road-specific mask-derived scaling
on the exposed private weak-prior projection.

The constant applied to the whole predictor gives relative L1 of 0.68357
and relative squared error of 0.54135. The road-specific scales give 0.68319
and 0.54086, versus 0.68934 and 0.54884 without scaling. Scaling the histogram
component alone yields most of the gain. Hidden OD accuracy is unverified.

From the experiment directory:

    python task4/audit_pairing_fixed_scale_control.py --release-root /path/to/kaggle_public --output /path/to/audit-output
