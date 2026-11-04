# Fixed OD measurements across covariance models

Both policies select locations under the joint origin and destination covariance. Batch conditioning evaluates those same locations under joint and diagonal covariance on one common unmeasured set. The other-cell policy leads under joint covariance at exponents 0 and 0.25. Under diagonal covariance it trails at every tested exponent. The covariance models and direct measurements are hypothetical.

    python task4/audit_direct_od_covariance_transfer.py --release-root /path/to/release --output /path/to/output
