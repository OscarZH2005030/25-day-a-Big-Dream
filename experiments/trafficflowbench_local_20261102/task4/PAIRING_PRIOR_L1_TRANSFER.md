# L1 calibration of the released prior pairing component

Fit a scalar shrinkage coefficient using normalized absolute error on validation roads from other corridor families. The weighted median solves the one-dimensional L1 fit. Compare the frozen coefficients on private released weak priors with the previous L2 fit and fixed coefficients.

With counts, margins and the span histogram projected out, the mean relative L1 errors are 0.89233 for L1 calibration, 0.89289 for L2 calibration, and 0.89256 for a fixed half coefficient. These experiments evaluate released weak priors.

    python task4/audit_pairing_prior_l1_shrink_transfer.py --release-root /path/to/release --output /path/to/output
