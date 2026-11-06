# Task4 frozen-design solver replay

For each replica, freeze its final 256 selected OD cells on I10 East and
rerun the minimum separation LP with HiGHS interior point. The automatic
and interior-point methods agree at each fixed design. Both produce
numerical zero for the first design and approximately 0.648 flow units
for the other. The change is therefore associated with the adaptive
design, and is not reproduced by changing the final solver alone.
These are hypothetical measurements.

    python task4/audit_pairing_od_probe_frozen_replay.py --release-root /path/to/kaggle_public --output /path/to/audit-output
