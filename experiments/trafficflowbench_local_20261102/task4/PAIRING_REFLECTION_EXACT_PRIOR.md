# Opposite pairing targets at exact prior distance

Preserve baseline counts, margins, the original weak-prior L1 distance and the signs of all deviations. Test both views, with and without the baseline span histogram. At 5% normalized L1 movement, both opposing targets can reach a gain of 0.05 in all 40 cases. Each move worsens the opposite target by 0.05. These are proxy optimization results under matched constraints.

    python task4/audit_pairing_reflection_exact_prior.py --release-root /path/to/release --output /path/to/output
