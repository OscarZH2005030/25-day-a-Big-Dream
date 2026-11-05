# Sign agreement in OD weak-prior transfer

Retain cells where the training and validation pairing components agree in sign, then reproject into the same nullspace. Compare the resulting half-average with the full half-average and the complementary component against the released private weak prior.

After projecting out counts, margins and span bins, sign gating increases relative L1 from 0.85784 to 0.87488 and relative squared error from 0.78074 to 0.81424. This is a released-prior transfer check.

    python task4/audit_pairing_prior_agreement_gate.py --release-root /path/to/release --output /path/to/output
