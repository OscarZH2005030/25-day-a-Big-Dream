# Group-error alignment after exact margins

Compare a group-constant additive error with a cell-scaled error for origins and destinations, holding each cell's full variance fixed. The group-constant amplitude uses the smallest cell variance in its group. This changes both the shape of the shared error and its total variance contribution. The experiment cannot attribute the difference to alignment alone.

At variance exponents 0.5 and 1, the group-constant model left more prior-weighted pairing variance in all 40 matched cases. At exponent zero both models coincide. The cell-scaled results were checked against a separate factor-space projection. These are covariance sensitivity results under hypothetical exact margins.

    python task4/audit_pairing_group_alignment.py --release-root /path/to/release --output /path/to/output
