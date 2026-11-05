# Selecting OD amplitude masks from one or two prior splits

Use train, validation or their equal average to rank cells by absolute pairing amplitude, then mask the same pooled prediction and match projected L2 norms. With counts, margins and span bins projected out at 5% retention, mean relative L1 is 0.87273 for the pooled mask, 0.88218 for train and 0.88336 for validation. Evaluation uses released private weak priors.

    python task4/audit_pairing_prior_mask_source.py --release-root /path/to/release --output /path/to/output
