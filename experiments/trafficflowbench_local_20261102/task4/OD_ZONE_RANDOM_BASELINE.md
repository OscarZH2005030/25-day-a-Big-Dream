# Random baseline for origin-total measurements

For each road and variance law, seed 20261104 draws 64 sets of 16 eligible origin totals uniformly without replacement within each set. The road-average gain is 23.0% to 23.5% for random selection and 28.9% to 32.9% for greedy selection. The comparison assumes exact hypothetical observations and conditional covariance.

    python task4/audit_od_zone_random_baseline.py --release-root /path/to/release --output /path/to/output
