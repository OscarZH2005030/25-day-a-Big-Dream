# Symmetric feasible movement toward a pairing proxy

Maximize normalized L1 movement aligned cellwise with the two-source pairing target, while allowing the same move in the opposite direction. Preserve baseline counts, origin and destination totals and exact original weak-prior L1 distance. The mean optimum is 6.09% of baseline demand without the span histogram fixed and 4.99% with it fixed. Both moves are checked directly. The values describe a feasible proxy movement, without hidden OD scores.

    python task4/audit_pairing_symmetric_radius.py --release-root /path/to/release --output /path/to/output
