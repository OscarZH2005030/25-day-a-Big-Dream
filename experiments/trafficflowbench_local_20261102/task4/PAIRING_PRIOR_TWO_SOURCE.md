# Fixed two-source shrinkage of released weak-prior pairing

Average the normalized train and validation priors in the private pairing subspace and apply a fixed half coefficient. Compare both sources separately and the full average. With the span histogram included in the projection, the half average yields relative L1 error 0.85784 versus 0.89256 for half of train and 0.89735 for half of validation. The target is the released private prior projection. The projection operates on signed vectors and does not establish feasibility of the implied OD flow.

    python task4/audit_pairing_prior_two_source.py --release-root /path/to/release --output /path/to/output
