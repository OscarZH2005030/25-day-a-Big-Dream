# Pairing value of an exact span histogram

On ten road networks and both validation and private count geometries, add a hypothetical exact span histogram to counts and exact origin and destination totals. Each case identifies one additional individual OD cell. Under an isotropic OD error model, the mean remaining unweighted variance decreases by 2.87%. Weighting OD errors by the released weak prior gives 1.82% in validation and 1.81% in private. The weights describe a sensitivity objective and the histogram is hypothetical.

    python task4/audit_pairing_exact_span_value.py --release-root /path/to/release --output /path/to/output
