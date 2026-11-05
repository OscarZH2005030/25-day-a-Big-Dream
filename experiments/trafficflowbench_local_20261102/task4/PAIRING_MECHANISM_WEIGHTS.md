# Task4 weights for specific disagreement types

The fixed multiplier is 0.7 on opposite signs, same-sign magnitude imbalance,
or both. Each predictor is projected back into the observation nullspace and
scaled to the uniform predictor norm. Scores are relative to zero prediction
on the same exposed private weak-prior projection.

With counts and margins, no rule improves both losses on any road. With the
span histogram added, the same-sign imbalance rule improves both on 7/10 roads.
Mean relative L1 moves from 0.85784 to 0.85485 and mean relative squared error
from 0.78074 to 0.77217. The histogram is a hypothetical constraint and changes
the target projection. Hidden OD performance remains unverified.

From the experiment directory:

    python task4/audit_pairing_mechanism_weights.py --release-root /path/to/kaggle_public --output /path/to/audit-output
