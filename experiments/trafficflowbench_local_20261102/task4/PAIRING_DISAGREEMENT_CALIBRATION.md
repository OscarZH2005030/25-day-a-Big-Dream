# Task4 cross-period disagreement calibration

For each road and each observation view, project normalized train, validation
and private released weak priors into the nullspace of the constraints. The
views hold counts and OD margins fixed, with or without the span histogram.
The fixed predictor is one half of the mean of the first two projections.

Compare disagreement below 0.5 with disagreement at or above 0.5 in each
quartile of the sum of the absolute train and validation projections.
Coordinates with source amplitude at most 1e-12 are excluded. Each group's
absolute and squared errors are divided by the respective error of predicting
zero in that same group. A positive paired delta means higher relative error
in the higher-disagreement group.

Both relative errors are higher in 38 of 40 road-by-quartile comparisons
under counts and margins, and 39 of 40 with the histogram. The comparisons
share roads and projections and should be read descriptively. This is
evidence about error against released weak priors. The target contains no
hidden OD labels, and the comparison does not estimate the benefit of
changing weights.

From the experiment directory:

    python task4/audit_pairing_disagreement_calibration.py --release-root /path/to/kaggle_public --output /path/to/audit-output
