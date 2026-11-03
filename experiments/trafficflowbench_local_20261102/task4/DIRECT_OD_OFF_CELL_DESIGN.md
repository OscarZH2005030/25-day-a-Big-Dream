# Direct OD measurements aimed at unselected cells

With sixteen hypothetical exact measurements, compare greedy total weighted covariance reduction with a rule that removes the directly measured cell's gain from each selection objective. Both rules use pooled historical weak-prior weights to choose cells and private weights to evaluate them. The geometry uses released counts only.

Across ten roads, the final reduction in unselected cells rises from 0.02366% to 0.04462% of the starting weighted trace. The total reduction falls from 13.31% to 0.61%. The gain on unselected cells remains small in absolute terms. This result is conditional on the assumed isotropic covariance and hypothetical exact observations. It does not estimate hidden OD scores.

    python task4/audit_direct_od_off_cell_design.py --release-root /path/to/release --output /path/to/output
