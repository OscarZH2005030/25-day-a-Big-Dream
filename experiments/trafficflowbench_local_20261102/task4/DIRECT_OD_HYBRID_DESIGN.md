# Mixing direct and indirect gains in OD measurement selection

Allocate sixteen hypothetical exact OD measurements. First allocate 0, 4, 8, 12 or 16 to greedy total gain, then allocate the remainder to gain on other cells. Choices use pooled historical weak-prior weights and evaluation uses private weak-prior weights. The geometry includes released counts only.

Across ten roads, allocating four measurements to total gain raises mean total trace reduction from 0.61% to 3.58%, with mean unselected-cell reduction moving from 0.04462% to 0.04537% of the initial weighted trace. Allocating all sixteen to total gain gives 13.31% total reduction and 0.02366% in unselected cells. The differences describe this isotropic covariance model under hypothetical observations and are not hidden OD scores.

    python task4/audit_direct_od_hybrid_design.py --release-root /path/to/release --output /path/to/output
