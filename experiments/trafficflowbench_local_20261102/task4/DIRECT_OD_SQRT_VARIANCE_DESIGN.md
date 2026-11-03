# Square-root OD variance sensitivity

This check keeps the sixteen-cell budget, count geometry, selection rules and evaluation weights fixed. It sets each independent pre-count variance proportional to the square root of its floored pooled historical weight. The square-root law lies between equal and linear variance weighting.

Across ten roads, other-cell selection reduces the unselected weighted trace by 0.487% of its initial total, compared with 0.227% for total-gain selection. Total reductions are 10.18% and 28.99%, respectively. The unselected-cell reductions are smaller than under the linear variance assumption. These are hypothetical covariance diagnostics. The variance law has not been estimated using independent OD errors.

    python task4/audit_direct_od_sqrt_variance_design.py --release-root /path/to/release --output /path/to/output
