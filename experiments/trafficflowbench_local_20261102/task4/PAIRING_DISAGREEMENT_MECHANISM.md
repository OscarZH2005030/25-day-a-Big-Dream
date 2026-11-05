# Task4 disagreement mechanisms

Hold the source amplitude bins and fixed half-average predictor constant.
Compare opposite-sign pairs and same-sign pairs with normalized disagreement
at least 0.5 to balanced same-sign pairs within each road, view and bin.

The gain from the predictor over zero is smaller on both loss measures in
364/400 opposite-sign comparisons and 189/399 same-sign imbalance comparisons.
One same-sign imbalance stratum is empty. These comparisons share roads and
use exposed weak-prior targets. They do not estimate the effect of reweighting
on hidden OD.

From the experiment directory:

    python task4/audit_pairing_disagreement_mechanism.py --release-root /path/to/kaggle_public --output /path/to/audit-output
