# Transferring hypothetical OD measurement choices

Select sixteen hypothetical direct OD measurements per road using either validation weak-prior weights or private weak-prior weights. Both selections share the same private count and hypothetical gross-margin geometry. Evaluate both with private weak-prior weights. Using private weights for selection is an in-split comparator. The private weak prior is a released input.

Across ten roads, validation-weight greedy selection retains 87.99% of private-weighted trace, versus 82.41% using private weights for selection. The two greedy choices share 7.1 of sixteen cells on average. Largest-prior selection retains 88.93% with validation weights and 82.94% with private weights. These numbers describe a hypothetical covariance under isotropic perturbations, with no new direct observations or hidden OD outcomes.

    python task4/audit_direct_od_measurement_transfer.py --release-root /path/to/release --output /path/to/output
