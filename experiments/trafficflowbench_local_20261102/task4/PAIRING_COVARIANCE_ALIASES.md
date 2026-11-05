# Count-invisible aliases of OD covariance

For each validation road, start with a fixed shared origin/destination covariance. Add an independent rank-one factor in the joint nullspace of counts and hypothetical exact margins. Choose the minimum and maximum train-prior-weighted variance among normalized projected OD coordinate directions. Both additions have equal total unweighted variance and are invisible to counts and margins.

The constructed pairs differ in weighted pairing variance by factors from 331.8 to 4521.5. This deliberately extreme comparison preserves total trace and changes individual cell variances. It shows that fitting observable covariance cannot identify the latent weighted variance, and does not estimate real error magnitudes.

    python task4/audit_pairing_covariance_aliases.py --release-root /path/to/release --output /path/to/output
