# Matching-only updates: benefit versus tail risk

This study uses 480 artificial cases, seed 2026110325, with known generated OD truth. It does not evaluate hidden competition labels or replace the frozen official reference.

The cross-input predictor removes the complete span/origin/destination feature space exactly and predicts held-out supplied prior entries with five-fold ridge regression. Direct methods mix that forecast into the supplied prior at fixed 5% or 25%, then solve the nonnegative count objective. Cycle methods start at the original baseline and approach the corresponding estimated direct candidate with feasible four-cell OD exchanges. Their target is an estimate, never generated truth.

Each accepted exchange preserves origin and destination margins and the original-prior L1 deviation within its original orthant. The release-specific incidence identity then preserves counts. Thus S_link, S_attr and S_dev stay unchanged, isolating the uncertain OD matching contribution. Random coordinate descent is a heuristic, not a proven global projection or optimum.

|Method|Mean artificial Task4 change|Worst change|Composite harms / 480|OD harms / 480|
|---|---:|---:|---:|---:|
|Direct 5%|+0.003273|-0.027749|55|63|
|Cycle 5%|+0.001043|-0.004336|81|81|
|Direct 25%|+0.011600|-0.181214|114|142|
|Cycle 25%|+0.002594|-0.034103|90|90|

This is a tradeoff, not dominance: cycle updates have lower mean benefits and smaller worst observed losses; at 5% they harm more cases. Both cycle variants move in all 480 cases. No method's OD score reaches its floor in this sample; unclipped relative OD L1 is also recorded. Preserving three score components does not establish the correct hidden pairing.

An independent readback checks all 2880 actions and 960 saved artifact hashes with direct NumPy scoring. Maximum score/movement error is 2.22e-16; maximum count/margin/prior-L1 drift is 7.28e-12. Numerical controls using oracle targets are confined to algorithm verification and never enter the tested estimator. See the result JSON for complete reproduction status; interrupted attempts are preserved separately and are not counted as successful reproduction.

Portable entry point:

```powershell
python task4/stress_od_cycles.py --release-root <release-root> --output <new-output-directory>
```

Neither weak-prior cross-validation nor these artificial outcomes prove hidden OD or official total-score improvement. The public data and mechanism families have already informed research choices; this is not a whole-system blind test. No Kaggle submission was made.
