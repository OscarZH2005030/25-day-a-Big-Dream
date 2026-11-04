# Balancing zone-total selection across variance laws

The balanced rule greedily maximizes the smallest fractional marginal trace reduction across three variance laws. All four rules select sixteen exact origin totals and are evaluated under each law. Across roads, the balanced rule has a mean worst gap of 2.75 percentage points relative to the best of the four rules per road and law. The square-root rule has a mean worst gap of 2.79 points and a lower maximum gap, 4.27 points versus 4.67. The comparison assumes hypothetical exact observations and covariance models.

    python task4/audit_od_zone_balanced_selection.py --release-root /path/to/release --output /path/to/output
