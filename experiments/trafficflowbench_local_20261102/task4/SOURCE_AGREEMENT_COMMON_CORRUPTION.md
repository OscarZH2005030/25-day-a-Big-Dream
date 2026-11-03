# Shared count-invisible error in two Task4 priors

Seed 2026110351 applies the same four-cell exchanges to both source priors across 80 reused artificial cases. The exchanges preserve both sources' loaded counts and gross origin and destination totals. Their L1 movement is capped at 10% of the smaller source total. Clean and perturbed variants use equal realized target movement in each case.

The agreement rule lost synthetic OD score in 39 of 80 cases and the mean rule in 32. The agreement rule had a worse mean score change in three of four noise families. These are within-rule changes under a simulated error. Absolute performance can rank differently. The single-source and shared-source tests use different seeded draws, so their harm counts cannot isolate the causal effect of sharing the same error.

    python task4/stress_source_agreement_common_corruption.py --release-root /path/to/release --output /path/to/output

The runner rebuilds the cases and verifies all 320 actions, including invariance and equal target movement. The data are reused synthetic cases with no hidden OD evaluation.
