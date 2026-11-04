# One evaluation mask for six OD measurement designs

Two objectives are crossed with three selection covariance rules. All six designs are evaluated on the OD cells that none measures, under two evaluation covariances and five variance exponents. Averaged over exponents and ten roads, the other-cell objective gives less reduction in all six selection and evaluation covariance combinations. All 600 full-cell reductions agree with earlier runs within 1e-12. This is a hypothetical measurement diagnostic.

    python task4/audit_direct_od_covariance_shared_mask.py --release-root /path/to/release --output /path/to/output
