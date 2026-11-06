# Task4 adaptive probe batch-size sensitivity on I10 East

Add 32 probes per round through 256, retaining the same constraints as
the eight-probe batch experiment. Both runs find indistinguishable pairs
through 224 probes. At 256, the minimum measured L1 differs across runs:
approximately zero locally and 0.648 flow units in the isolated rerun.
The full OD witness gaps are 0.505% and 0.496% of demand respectively.
Adaptive selection can vary with solver ties, so the two runs need not
test the same coordinates. These results do not establish that increasing
batch size reliably separates the feasible sets.

    python task4/audit_pairing_od_probe_batch32.py --release-root /path/to/kaggle_public --output /path/to/audit-output
