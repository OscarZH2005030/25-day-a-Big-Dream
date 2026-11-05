# Task4 joint matching across ten roads

Extend the two-road audit to the remaining eight at a 1% total movement cap.
Across all ten, matching both normalized squared span and median-threshold
long-trip share requires 2.44% to 5.82% slack above each problem's minimax
optimum. Both runs reproduce the minimum slack. Common values at the optimum
can differ when several matching flows attain the same threshold. The solver
checks that both flows match within each run. Hidden OD accuracy is unverified.

From the experiment directory:

    python task4/audit_pairing_minimax_joint_extension.py --release-root /path/to/kaggle_public --output /path/to/audit-output
