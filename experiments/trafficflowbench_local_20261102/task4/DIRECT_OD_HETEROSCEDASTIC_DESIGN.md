# Sensitivity of direct OD measurement design to unequal variances

This comparison keeps the released count rows, the two selection objectives, the sixteen-measurement budget and the evaluation weights fixed. It changes the assumed independent pre-count variances from equal to proportional to pooled historical weak-prior flows, with a floor of 10% of their mean.

Across ten roads, the weighted trace reduction in unselected cells is 4.29% with other-cell selection and 1.61% with total-gain selection. The respective total reductions are 17.87% and 40.90%. These numbers are percentages of the starting trace within the unequal-variance model. The variance law is a sensitivity assumption. No actual OD measurement or hidden score is used.

    python task4/audit_direct_od_heteroscedastic_design.py --release-root /path/to/release --output /path/to/output
