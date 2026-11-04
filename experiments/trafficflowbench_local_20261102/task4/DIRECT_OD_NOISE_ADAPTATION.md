# Noise-aware OD measurement selection

Four designs cross two objectives with selection noise variance 0 or 0.25. All four are scored with variance 0.25, using the same set of unmeasured cells for the indirect metric. Including noise in selection changes the other-cell policy's average overall reduction by 0.20 to 0.51 percentage points across the five variance laws. Its change on the common set is smaller than 0.013 points in magnitude. These are covariance diagnostics under hypothetical measurement and variance assumptions.

    python task4/audit_direct_od_noise_adaptation.py --release-root /path/to/release --output /path/to/output
