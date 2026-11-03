# Reducing invisible L1 with the other score components fixed

Seed 2026110346 covers forty artificial cases at count-shape TV 0.14. A nonnegative LP minimizes the invisible L1 correction while preserving baseline counts, gross margins, original-prior orthant and original-prior L1 distance. The study compares the baseline, 5% interpolation and the full LP solution. The three non-OD components stay fixed.

|Artificial mechanism|Blend|Mean OD change|OD harms / 20|
|---|---:|---:|---:|
|Shared lognormal|5%|+0.001020|0|
|Shared lognormal|100%|-0.002823|11|
|Shared positive background|5%|-0.000210|7|
|Shared positive background|100%|-0.009606|16|

Small shrinkage helps the sampled lognormal cases. It fails in the shared-background mechanism, and full shrinkage has negative means in both. The fixed LP objective alone does not order OD accuracy. These artificial mechanisms and networks do not identify an official replacement.

Independent readback verified all 120 actions, unchanged non-OD components and the composite change identity. The portable rerun reproduced four result files and all forty case hashes.

```powershell
python task4/stress_invariant_invisible_shrink.py --release-root <release-root> --output <new-output-directory>
```
