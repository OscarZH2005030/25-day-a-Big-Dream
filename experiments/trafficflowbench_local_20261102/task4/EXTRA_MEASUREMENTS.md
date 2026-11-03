# OD information from hypothetical trip-span observations

Zone-index span is destination index minus origin index. Total span is already fixed by the two OD margins. On the twenty network/input combinations, a full span histogram adds 33 to 98 independent linear measurements. Adding the origin- and destination-specific span sums brings the combined increment to 96 to 291, leaving linear nullity between 465 and 4,560 before nonnegativity and prior constraints.

Seed 2026110341 fixes one objective per network/input pair. All 40 LPs find a nonnegative flow preserving counts, margins, original-prior L1 distance and the specified hypothetical measurements. Relative OD L1 movement is 0.231 to 0.436 with the histogram and 0.165 to 0.247 with both the histogram and marginal span sums. These are feasible witnesses from fixed objectives. They do not bound the largest possible OD change.

Independent SVD checks all 80 ranks. Independent readback checks all 40 flow constraints. The portable runner reconstructs the anchor from the frozen prediction CSV, reproduces the scientific result fields and matches all 40 witness hashes. The anchor provenance field differs because the portable run does not require the earlier LP witness file.

These are measurement-design diagnostics on released geometry. Zone-index span is not a measured trip distance. No hypothetical observation enters the official baseline.

```powershell
python task4/audit_extra_measurements.py --release-root <release-root> --output <new-output-directory>
```
