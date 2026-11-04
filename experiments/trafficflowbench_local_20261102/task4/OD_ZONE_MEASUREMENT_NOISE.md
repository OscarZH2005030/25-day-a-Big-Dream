# Noisy measurements of fixed origin totals

The sixteen locations for each policy are fixed from exact-measurement selection. Batch conditioning evaluates those sets after adding independent measurement noise at 0.25 and 1 times the mean eligible marginal variance. This isolates the loss from measurement noise without changing the locations or using hidden labels.

    python task4/audit_od_zone_measurement_noise.py --release-root /path/to/release --output /path/to/output
