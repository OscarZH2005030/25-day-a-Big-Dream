# Sensitivity to a count-invisible source error

Seed 2026110350 applies nonnegative OD exchanges to one source prior in each of 80 reused synthetic cases. The exchanges preserve its loaded link counts and gross origin and destination totals, with a 10% relative L1 cap. The target prior, synthetic truth, and baseline stay fixed. The clean and corrupted fits use equal realized target movement within each case.

The agreement rule's synthetic OD score fell in 39 of 80 cases after corruption. The mean rule's score fell in 24. Agreement had a worse average corruption response in three of the four noise families, and its fitted OD vector moved more on average in those same families. Under independent lognormal noise, agreement was less sensitive by both averages. These are perturbation responses within each rule. They do not establish that the mean rule has a higher absolute score.

Run with the benchmark release root and a writable output directory.

    python task4/stress_source_agreement_corruption.py --release-root /path/to/release --output /path/to/output

The runner regenerates the 80 artificial cases and checks scores, hashes, nonnegativity, count and margin invariance, prior L1 invariance, and matched movement. It writes aggregate prediction sensitivity alongside the score tables. The perturbation is artificial and the cases are reused. There is no hidden OD validation or new official score.
