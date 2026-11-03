# Pairing drift with identical counts and margins

Seed 2026110340 pairs 40 artificial moderate-drift cases with versions in which bounded four-cell exchanges rearrange target OD flow. The construction preserves target counts and both margins. It keeps source priors, error draws and folds fixed. The target supplied prior is regenerated for the changed truth and recalibrated to count-shape TV 0.14. The resulting scenarios share counts and source information while their target priors can differ. Relative target L1 movement ranges from 0.518 to 0.761.

|Artificial mechanism|Target version|Clipped mean Task4 change|Composite harms / 20|OD harms / 20|
|---|---|---:|---:|---:|
|Shared lognormal|Moderate|+0.003312|3|0|
|Shared lognormal|Re-paired|+0.003611|1|0|
|Shared positive background|Moderate|+0.001087|6|7|
|Shared positive background|Re-paired|-0.002793|20|19|

The shared-background result changes sign after re-pairing while the lognormal mean stays positive. The raw source fit also deteriorates under shared background. This isolates a risk that target count and marginal checks cannot reveal in the specified synthetic construction. The stress uses previously exposed networks and does not identify the true hidden error mechanism.

Independent readback verified 240 action scores, both margins, counts and exact paired source inputs and folds. The portable rerun matched both tables, the empty failure list and all 80 case artifact hashes.

```powershell
python task4/stress_count_invariant_drift.py --release-root <release-root> --output <new-output-directory>
```
