# Transfer of the released weak prior pairing component

Project normalized training and evaluation priors into the joint count and margin nullspace, with and without a hypothetical exact span histogram. The evaluation geometry is available from the released path and count maps. The target is a released weak prior component.

On private with the histogram included, full transfer has mean squared error 1.1005 times the zero-pairing comparator. A coefficient fitted on validation projections from other corridor families reduces it to 0.8274, while a fixed half coefficient gives 0.8297. Without the histogram, the corresponding ratios are 0.6107, 0.5337, and 0.5722. These are repeatability checks on released priors, without hidden OD validation.

    python task4/audit_pairing_prior_repeatability.py --release-root /path/to/release --output /path/to/output
    python task4/audit_pairing_prior_shrink_transfer.py --release-root /path/to/release --output /path/to/output
