# Task4 histogram preservation after scaling

Compare scaling the full prediction, scaling only the histogram-nullspace
component, and leaving the same correction unscaled. The last two preserve
the baseline prediction's span histogram by construction. This compares
hypothetical projected predictions against exposed weak-prior targets.

Scaling only the nullspace component lowers relative squared error from
0.54884 to 0.54549 and raises relative L1 from 0.68934 to 0.68988. Leaving the
correction unscaled yields 0.54855 and 0.68916. The larger simultaneous
improvement under whole-prediction scaling allows the histogram to move.
Hidden OD performance remains unverified.

From the experiment directory:

    python task4/audit_pairing_histogram_preserving_norm.py --release-root /path/to/kaggle_public --output /path/to/audit-output
