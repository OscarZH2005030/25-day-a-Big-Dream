# Cross-model OD measurement selection

Four designs choose sixteen hypothetical exact OD measurements using pooled historical weak-prior weights. Three optimize one variance law and the fourth maximizes the worst relative marginal gain at each step. Each fixed selection is evaluated under all three variance laws with private weak-prior weights.

Relative to the best tested design within each road and law, the balanced design has a mean per-road worst gap of 0.495 percentage points. The linear-variance design has a gap of 0.140 percentage points. Its largest gap across roads and laws is 1.043 percentage points, compared with 1.863 for the balanced design. These are trace reduction diagnostics under assumed covariances and hypothetical exact measurements.

    python task4/audit_direct_od_variance_transfer.py --release-root /path/to/release --output /path/to/output
