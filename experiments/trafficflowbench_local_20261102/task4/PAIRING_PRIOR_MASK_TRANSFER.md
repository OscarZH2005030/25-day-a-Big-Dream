# Transfer of OD amplitude masks across observation assumptions

Compare selecting the largest OD amplitudes in the evaluation subspace against selecting them in the other subspace, with or without projecting out a span histogram. Keep selection size and projected L2 norm fixed. At 5% retention, the margin-only mask gives relative L1 of 0.89150 in the histogram view, compared with 0.87273 for selection in that view. The target is the released private weak-prior projection.

    python task4/audit_pairing_prior_mask_transfer.py --release-root /path/to/release --output /path/to/output
