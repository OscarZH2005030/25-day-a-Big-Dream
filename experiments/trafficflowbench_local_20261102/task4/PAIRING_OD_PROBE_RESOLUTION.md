# Task4 direct OD probe resolution

On two pilot roads, take two fixed feasible OD witnesses that match
counts, margins and span histograms. Place probes on the 1, 4 and 8
OD cells with the largest absolute differences. For independent equal
Gaussian observation errors and equal prior odds, the optimal classifier
has accuracy Phi(signal_L2 / (2 * sigma)).

The maximum per-probe noise SD for 95% accuracy is about 0.37 to 0.67
flow units for one probe and 1.04 to 1.81 for eight probes. These are
optimistic requirements for the two particular witnesses and selected
cells. There are no real observations in this test, and the thresholds
give no evidence about which OD solution is closer to hidden demand.

    python task4/audit_pairing_od_probe_resolution.py --release-root /path/to/kaggle_public --output /path/to/audit-output
