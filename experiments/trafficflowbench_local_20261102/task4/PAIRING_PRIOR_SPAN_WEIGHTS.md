# Fixed span weights in OD prior transfer

Compare uniform pooling, short-band weight 0.7 and other-band weight 1.3, and the reverse. Reproject and match the norm of the uniform prediction. Reducing short-band weight changes mean relative L1 from 0.68934 to 0.68475 with counts and margins removed, and from 0.85784 to 0.85965 with the span histogram also removed. The evaluation targets are released weak priors.

    python task4/audit_pairing_prior_span_weights.py --release-root /path/to/release --output /path/to/output
