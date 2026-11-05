# Boundary check for released residual covariance

Expand the original grid to include diagonal covariance and shared fractions up to 0.99. In all 20 road/view comparisons the train-selected shape and the retrospective validation optimum coincide at power zero, shared fraction 0.99, and equal origin/destination shares. The optimum remains at the tested boundary.

Train and validation released counts fall within the linear image of the path incidence matrices to floating-point tolerance. This check does not establish nonnegative feasibility or a particular residual-generation mechanism. Scores remain a count-space proxy.

    python task4/audit_released_residual_boundary.py --release-root /path/to/release --output /path/to/output
