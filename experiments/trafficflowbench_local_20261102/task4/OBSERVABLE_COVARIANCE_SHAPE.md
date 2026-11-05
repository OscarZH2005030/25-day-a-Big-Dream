# Count-space distinguishability of OD error covariance

Compare the count-space Gaussian covariances implied by three origin shares with the same cell variances, shared fraction and variance exponent. Use directional KL divergence for one hypothetical centered count vector. The comparison covariance's global scale minimizes expected KL. No scale was fitted to an observed count-error vector.

The 360 directional comparisons span 1.91 to 117.19 nats after scale optimization. This measures distinguishability under the assumed Gaussian families and known mean. It does not establish that released counts come with observed centered errors or that real covariance parameters can be recovered.

    python task4/audit_observable_covariance_shape.py --release-root /path/to/release --output /path/to/output
