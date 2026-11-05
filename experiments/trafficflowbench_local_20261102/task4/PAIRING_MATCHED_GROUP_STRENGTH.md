# OD pairing variance with matched shared error strength

Match total variance for every OD cell and pooled-prior-weighted shared variance across both error models on each road. Redistribute independent variance to keep cell totals fixed. Compare cell-scaled and group-constant origin and destination factors under exact counts and hypothetical exact margins.

With heterogeneous cell variances, the group-constant law retained more weighted OD pairing variance in all 40 paired cases. Both laws coincide in the 20 uniform-variance comparisons. The normalization holds shared strength fixed in aggregate. The cellwise allocation of shared variance and correlation pattern can both differ. These sensitivities do not validate hidden OD estimates.

    python task4/audit_pairing_matched_group_strength.py --release-root /path/to/release --output /path/to/output
