# Source scale is not the main explanation in this control

The earlier source-count recipe changed both the source flows and their positive-median normalization. This new registered experiment isolates that confound with seed 2026110328: 120 fresh artificial draws, ten released networks, four demand families and three drift regimes, restricted to the previously problematic shared positive background mechanism. The strength is fixed at 5%; no case or strength is selected for deployment.

|Source features|Mean artificial Task4 change|Worst change|Composite harms / 120|OD harms / 120|
|---|---:|---:|---:|---:|
|Original priors|-0.00207534|-0.02844506|37|36|
|Count estimates, own positive-median scale|-0.00233193|-0.02840769|79|73|
|Count estimates, original prior scale|-0.00233104|-0.02840721|79|73|

Holding the original source scale fixed barely changes this result. Both count-adjusted recipes still harm more cases than using original sources. This narrows one implementation confound; it does not establish that all possible source-count methods fail or that normalization never matters. All three recipes have negative mean change in this particular stress family.

The generator produces source and target truths separately from observations. Learners receive only geometry, supplied priors, source counts and fixed folds. Each source correction uses its own counts. Target predictions share five folds and training-only target scale; scoring always measures deviation from the original supplied prior. Truth, inputs, fitted coefficients and predictions remain local, with aggregate results and hashes published.

Portable reproduction:

```powershell
python task4/stress_source_scale.py --release-root <release-root> --output <new-output-directory>
```

See `results/task4_source_scale_control/` for independent score readback and full reproduction status. This is a follow-up to an adaptively studied mechanism, not a whole-system blind evaluation or evidence about official hidden OD scores. The official Task4 reference remains frozen. No Kaggle upload was made.
