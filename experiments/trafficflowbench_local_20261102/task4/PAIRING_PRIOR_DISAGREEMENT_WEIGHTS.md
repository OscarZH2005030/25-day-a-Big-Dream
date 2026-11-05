# Continuous disagreement weights for OD prior transfer

Compare fixed downweighting and upweighting of normalized disagreement between the train and validation pairing components. Reproject and match the uniform prediction's L2 norm. With counts and margins projected out, downweighting raises both errors. With span bins also projected out, it reduces relative L1 from 0.85784 to 0.85644 while raising relative squared error from 0.78074 to 0.78120. These are mixed results against released weak priors.

    python task4/audit_pairing_prior_disagreement_weights.py --release-root /path/to/release --output /path/to/output
