# Fixed OD pairing sparsity budgets

Retain the largest 5%, 10%, 25%, 50%, 75%, or 100% of half-averaged source amplitudes before reprojecting to the count and margin nullspace. Match L2 norm to the ungated prediction. With the span histogram projected out as well, the 5% mask gives relative L1 of 0.87273 versus 0.85784 at 100%. Without projecting out the histogram, the corresponding errors are 0.75577 and 0.68934. All budgets are reported against released weak priors.

    python task4/audit_pairing_prior_sparsity_curve.py --release-root /path/to/release --output /path/to/output
