# Task4 sensitivity to linear-program solution selection

The primary program matches either the half-average or fixed component
target within a 5% L1 movement budget. The secondary program keeps primary
target L1 within 1e-8 of optimal relative to demand and minimizes the largest
absolute cell movement. Both preserve nonnegativity, frozen counts and
margins, and the bound on original prior distance. The second view also
fixes the span histogram.

The two replicas agree on primary objective gains, movement, and minimax
objective values within 1e-10 of total demand. Solutions can still differ:
the exposed prior errors are recorded separately for both replicas. In
both runs with counts and margins, the fixed component target has lower
mean squared proxy error under the primary selection, and higher mean
squared proxy error under minimax selection. This is evidence of sensitivity
to the choice of feasible OD solution. Hidden OD accuracy is unverified.

From the experiment directory:

    python task4/audit_pairing_component_minimax.py --release-root /path/to/kaggle_public --output /path/to/audit-output
