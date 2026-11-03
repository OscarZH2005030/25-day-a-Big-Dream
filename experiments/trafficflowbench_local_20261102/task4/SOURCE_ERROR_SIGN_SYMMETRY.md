# Task4 source error alignment under both signs

The 80 reused artificial cases use the same feasible exchanges as the matched source error study. Each case compares all four source sign combinations. The ten target fits share one realized movement budget, and each alignment group averages over both signs.

At that common budget, the source agreement rule's aligned-minus-opposed mean is negative in all four noise families when the first sign is positive. It is negative in one family when that sign is negative. Averaging both signs gives two negative and two positive means, between -7.41e-6 and +9.07e-6. The earlier one-sided result depends on the perturbation direction. The mean rule's four symmetric contrasts are positive on these cases. These contrasts are sensitivity estimates on reused artificial cases with no hidden OD validation.

    python task4/stress_source_error_sign_symmetry.py --release-root /path/to/release --output /path/to/output

The verifier checks 80 artifacts and 800 fits, including the matched budget, both gross margins, link counts, and source nonnegativity under both signs.
