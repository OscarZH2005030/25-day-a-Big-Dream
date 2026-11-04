# Reverse covariance transfer for OD measurements

The two policies choose locations under diagonal covariance. Both sets are evaluated under diagonal and joint origin-destination covariance. In each of the five variance laws, the other-cell objective gives a smaller mean covariance reduction on the common unmeasured cells under either evaluation covariance. This experiment uses hypothetical covariance and exact measurements.

    python task4/audit_direct_od_reverse_covariance_transfer.py --release-root /path/to/release --output /path/to/output
