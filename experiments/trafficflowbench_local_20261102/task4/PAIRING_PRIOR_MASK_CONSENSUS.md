# Comparing two-source amplitude rankings

Rank OD cells by absolute signed pooled amplitude or by the minimum, maximum, or mean of the two source magnitudes. Apply each mask to the same pooled vector and match its projected norm. At 5% retention with margins and span bins projected out, the relative L1 errors are 0.87273 for signed pooling, 0.87404 for mean magnitude, 0.87479 for maximum magnitude and 0.88469 for minimum magnitude. Evaluation uses released private weak priors.

    python task4/audit_pairing_prior_mask_consensus.py --release-root /path/to/release --output /path/to/output
