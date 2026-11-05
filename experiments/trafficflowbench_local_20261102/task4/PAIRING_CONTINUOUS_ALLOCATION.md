# Continuous bounds on shared OD error allocation

The conditional pairing trace is concave in the allocation between origin and destination factors. For each fixed linear predictor its residual variance is affine in the allocation, and the conditional trace minimizes over those predictors. The minimum is therefore at an endpoint. A whitened eigendecomposition of the constraint covariance allows an analytic derivative and locates the maximum without choosing a finer grid.

The 300 reproduced grid values match the prior experiment. The worst allocation in the 40 heterogeneous cases assigns 45.50% to 57.48% to origins. Its pairing variance is at most 0.0292% higher than equal allocation. Uniform variances give a flat curve. The margins are hypothetical and the covariance laws are sensitivity assumptions.

    python task4/audit_pairing_continuous_allocation.py --release-root /path/to/release --output /path/to/output
