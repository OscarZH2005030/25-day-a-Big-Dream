# Sign gating at matched projection norm

Rescale the sign-gated projection to the L2 norm of the full half-average on each road and view. After projecting out counts, margins and span bins, relative L1 is 0.87236 for the matched gate and 0.85784 for the full half-average. Relative squared errors are 0.81194 and 0.78074. Targets are released weak priors.

    python task4/audit_pairing_prior_gate_matched_energy.py --release-root /path/to/release --output /path/to/output
