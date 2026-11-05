# Synthetic recovery of origin and destination error allocation

Generate 512 centered Gaussian count draws for each of 180 road and covariance combinations, using seed 2026110501. Profile a scale separately for each candidate and compare selection with the power and shared fraction fixed or searched on the registered grid. Both scoring regimes use identical draws.

Mean allocation accuracy is 98.03% with fixed nuisance shape and 97.15% with nuisance shape searched. The lowest accuracy in the latter regime is 67.38%. Original count rows fix the simulation coordinates across numerical environments. These draws have the correct center, and the candidate grid contains their generating covariance. The result does not establish empirical covariance recovery or hidden OD accuracy.

    python task4/audit_synthetic_allocation_recovery.py --release-root /path/to/release --output /path/to/output
