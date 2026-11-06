# Task4 eight-probe stability near the minimum OD gap

Choose the eight largest differences in one computed witness pair and
then minimize the sum of absolute differences at those fixed coordinates
over pairs with total OD gap within 1e-8 of its minimum relative to demand.
On the two roads, both runs retain over 99% of the selected witness signal.
The minimized sums are approximately 16.83 and 9.61 flow units, with
small differences across runs. Probe selection can depend on the witness
and numerical ties. This result covers only the restricted feasible
pairs in the experiment. It uses no actual OD observations.

    python task4/audit_pairing_od_probe_stability.py --release-root /path/to/kaggle_public --output /path/to/audit-output
