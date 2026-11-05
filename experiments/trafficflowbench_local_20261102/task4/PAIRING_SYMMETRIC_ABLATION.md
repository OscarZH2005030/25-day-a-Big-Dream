# Constraints on symmetric OD movement

Preserve baseline count predictions and origin and destination totals. Compare the symmetric movement radius with nonnegative flows, with the original weak-prior deviation signs also fixed, and with exact total L1 deviation added. The movement is aligned cell by cell with the two-source pairing proxy. With the span histogram fixed, mean radii are 16.98%, 5.08%, and 4.99% of baseline total demand. The three feasible sets are nested. These experiments use frozen baseline constraints.

    python task4/audit_pairing_symmetric_ablation.py --release-root /path/to/release --output /path/to/output
