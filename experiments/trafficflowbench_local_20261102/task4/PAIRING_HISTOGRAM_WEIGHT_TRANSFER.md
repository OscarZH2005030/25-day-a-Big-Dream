# Task4 histogram mask transfer

Derive the fixed same-sign imbalance mask either with or without projection
against the span histogram. Apply each mask to the same counts-and-margins
half-average, reproject, and match its L2 norm. Evaluate both against the
same counts-and-margins projection of the exposed private weak prior.

The histogram-derived mask lowers mean relative squared error from 0.54884
to 0.54559 and raises mean relative L1 from 0.68934 to 0.68985. The mask
derived directly from counts and margins raises both errors. The result
does not validate the histogram as an observation or show a hidden OD gain.

From the experiment directory:

    python task4/audit_pairing_histogram_weight_transfer.py --release-root /path/to/kaggle_public --output /path/to/audit-output
