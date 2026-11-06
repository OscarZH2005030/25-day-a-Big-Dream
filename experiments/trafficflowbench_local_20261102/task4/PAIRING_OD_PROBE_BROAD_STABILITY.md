# Task4 eight direct probes on a wider feasible set

Select the eight largest OD-cell differences of a minimum-gap witness.
Retain both primary objective tolerances, the shared complete span
histogram, 1% total movement and 125% maximum-cell caps. Allow the
total OD gap to vary.

Both runs find two feasible solutions with equal measurements on all
eight selected cells for each road. The full OD L1 differences across
runs are 0.929% to 0.938% and 0.835% to 0.840% of demand. The witnesses
can vary. The result applies to these probe designs and feasible sets
and uses no actual OD observations.

    python task4/audit_pairing_od_probe_broad_stability.py --release-root /path/to/kaggle_public --output /path/to/audit-output
