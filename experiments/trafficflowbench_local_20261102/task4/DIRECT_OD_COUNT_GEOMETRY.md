# Why two count-only direct OD selection rules agree

The count-row Gram pseudoinverse independently reconstructs the isotropic covariance conditioned on released counts. Across ten roads and four weight sources, greedy weighted-trace selection and descending weak-prior weights select the same sixteen OD cells in the same order in all forty comparisons.

The average share of greedy gain from cells other than the one selected at that step ranges from 0.10% to 0.12% across weight sources. The design mostly reduces uncertainty in directly measured cells. It assumes hypothetical exact direct OD measurements and isotropic perturbations. No hidden OD values or official score changes are inferred.

    python task4/audit_direct_od_count_geometry.py --release-root /path/to/release --output /path/to/output
The two result tables reproduced within 4.1e-16 absolute error. Selections, protocol and input hashes matched exactly.
