# Fixed OD measurement locations under noise

Both policies choose sixteen locations assuming exact measurements. Batch covariance conditioning then evaluates those same locations at noise variances 0, 0.25 and 1. On their shared unmeasured cells, the mean reduction is lower for the other-cell policy at every tested noise level and variance law. Both policies lose reduction as noise increases. The zero-noise scores agree with sequential conditioning within 1e-12.

    python task4/audit_direct_od_fixed_noise.py --release-root /path/to/release --output /path/to/output
