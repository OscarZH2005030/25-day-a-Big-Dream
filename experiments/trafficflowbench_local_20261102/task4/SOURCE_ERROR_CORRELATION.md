# Matched Task4 source error alignment

The experiment reuses 80 artificial cases and the exchanges generated with seed 2026110351. It shrinks each exchange until both signs are feasible for both source priors. It then compares (+,+) and (+,-) while holding the perturbation magnitude, loaded counts and gross margins fixed. Within each case, all six target fits use the same realized movement.

For the source agreement rule, aligned errors gave lower mean synthetic OD scores than opposed errors in all four noise families. The aligned-minus-opposed means range from -1.56e-5 to -2.63e-6. Mean source perturbation amplitudes range from 0.12% to 2.64% by noise family. The comparison holds the first source's perturbation sign fixed. It does not average over reversal of both signs.

    python task4/stress_source_error_correlation.py --release-root /path/to/release --output /path/to/output

The verifier checks all 480 fits and 80 artifacts. This experiment measures sensitivity on reused artificial cases. It contains no hidden OD labels or official score estimate.
