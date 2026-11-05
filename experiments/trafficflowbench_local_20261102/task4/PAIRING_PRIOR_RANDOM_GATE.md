# Random mask controls for OD pairing transfer

Compare the sign and amplitude masks with 32 uniform random masks per road and view, matched for the number of retained cells and projected norm. After removing the count, margin and span components, the magnitude gate beats all 320 matched random masks on relative L1. Its mean error is 0.85927 versus 0.90420 for random and 0.85784 without a mask. The targets are released weak-prior projections.

    python task4/audit_pairing_prior_random_gate.py --release-root /path/to/release --output /path/to/output
