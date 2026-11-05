# OD pairing gates at matched size and norm

Select the same number of cells as the sign-agreement gate by ranking absolute half-average amplitudes. Reproject both masks and match their L2 norms to the ungated prediction. With counts, margins and span bins projected out, relative L1 error is 0.85927 for the magnitude gate, 0.87236 for sign agreement and 0.85784 without a gate. Targets are released weak-prior projections.

    python task4/audit_pairing_prior_magnitude_gate.py --release-root /path/to/release --output /path/to/output
