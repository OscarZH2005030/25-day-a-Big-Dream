# Feasibility of two-source pairing shrinkage

The signed target combines baseline observable quantities with a half-shrunk train/validation pairing projection. Direct movement toward that target stops at step zero on all ten roads in both views. A nearest nonnegative L1 projection preserves baseline modeled counts and margins, with an optional exact span histogram. With that histogram its mean movement from baseline is 55.5% of total demand in normalized L1, and its mean distance to the released prior increases by 35.5% of total demand. The reproduced objective matches within 7e-16. Alternate LP optima change normalized movement by up to 0.00710 and prior distance by up to 0.00639 per road. Both runs are retained. These are feasibility and proxy results.

    python task4/audit_pairing_two_source_feasibility.py --release-root /path/to/release --output /path/to/output
