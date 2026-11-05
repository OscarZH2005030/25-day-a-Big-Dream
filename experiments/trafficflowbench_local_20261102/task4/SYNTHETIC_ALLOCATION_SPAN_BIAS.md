# Synthetic allocation recovery with span-dependent mean bias

Reuse 92,160 seeded Gaussian draws. Shift the count mean along the prior-weighted OD span profile and compare raw likelihood, removing the prior count direction, and removing both the prior and span directions.

At the largest shift, mean allocation recovery is 87.24%, 90.57%, and 96.80%, respectively. With no shift these are 97.15%, 96.97%, and 96.80%. The span direction is specified in advance and is the direction used to generate the synthetic shift. These results do not establish the real error direction.

    python task4/audit_synthetic_allocation_span_bias.py --release-root /path/to/release --output /path/to/output
