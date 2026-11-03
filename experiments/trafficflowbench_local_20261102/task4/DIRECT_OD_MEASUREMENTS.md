# Choosing hypothetical direct OD measurements

This audit starts from the public counted-link matrix and adds hypothetical exact gross origin and destination totals. It projects isotropic OD perturbations into the joint null space. A direct measurement conditions one OD cell exactly. The comparison allocates four or sixteen measurements using seeded uniform selection, largest weak-prior flow, or greedy reduction of the weak-prior-weighted covariance trace.

Across the ten private panels, sixteen greedy choices reduce the average weighted trace by 17.59%. Choosing the largest prior flows reduces it by 17.06%, and uniform selection by 0.92%. The unweighted greedy trace falls by only 0.81%. This design targets uncertainty under one assumed weighting and covariance. It leaves most OD dimensions unresolved, and the direct measurements and gross margins are hypothetical. It supplies no hidden OD score or submission change.

    python task4/audit_direct_od_measurements.py --release-root /path/to/release --output /path/to/output

The entry point regenerates all 400 panel rows and checks them with covariance downdates. It requires the public release network, counts and weak priors. No direct OD labels are required.
