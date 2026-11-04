# Comparing OD policies on the same unmeasured cells

The earlier comparison measured indirect gain over a different set of cells for each policy. Here both balanced policies are scored on the intersection of their unmeasured cells, with the same full initial trace as denominator. Each policy measures sixteen cells, with a mean overlap of 2.2 cells across ten roads.

For all five variance laws, the mean common-set reduction is smaller under the policy prioritizing other cells. The difference ranges from about 0.002 to 0.452 percentage points. All 100 original metric rows match the earlier studies within 1e-12. These results describe covariance under hypothetical exact measurements.

    python task4/audit_direct_od_common_support.py --release-root /path/to/release --output /path/to/output
