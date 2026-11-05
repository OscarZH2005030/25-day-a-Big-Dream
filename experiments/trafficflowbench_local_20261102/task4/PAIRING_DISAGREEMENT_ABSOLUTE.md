# Task4 absolute-error control

The fixed half-average predictor and amplitude quartiles match the earlier
disagreement calibration. This audit compares per-cell mean absolute error
and mean squared error, plus the gain from the half-average relative to
zero. Each paired delta subtracts the lower-disagreement group from the
higher-disagreement group in the same road, constraint view and quartile.

Higher disagreement has higher absolute and squared error together in 33/40
comparisons with counts and margins and 35/40 with the histogram added.
Gain over zero is smaller on both metrics in 33/40 and 40/40, respectively.
Source amplitude can still differ within a quartile, especially the largest
quartile. These descriptive comparisons use exposed weak-prior targets.

From the experiment directory:

    python task4/audit_pairing_disagreement_absolute.py --release-root /path/to/kaggle_public --output /path/to/audit-output
