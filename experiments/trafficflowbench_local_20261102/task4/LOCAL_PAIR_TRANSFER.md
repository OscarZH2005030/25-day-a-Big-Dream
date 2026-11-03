# Same-pair signal and neighboring-pair controls

The audit asks whether the cross-input signal extends to nearby origin-destination pairs with the same span. For each target input, the other two priors supply a geometry-orthogonal residual. The three controls use the residual at the same pair, at adjacent origins within the same span, or after a half-group cyclic shift within the span. Groups shorter than four are zeroed before re-projection. Re-projection can mix source entries.

Seed 2026110336 fixes five shared target folds per network. All fits use ridge 1, the same geometry and training-fold target median. The outcome is normalized prediction error on supplied weak-prior entries.

|Target input|Same-pair MSE reduction|Adjacent-span reduction|Distant-span reduction|
|---|---:|---:|---:|
|Train|20.16%|-1.96%|-0.15%|
|Validation|18.80%|-0.94%|-1.44%|
|Private public input|17.72%|-2.50%|-0.67%|

In these public inputs the gain is specific to the same-pair feature. Adjacent origins do not recover it with this fixed transformation. The audit does not separate persistent true matching structure from persistent prior error. It uses exposed supplied priors and has no hidden OD labels.

Independent readback verified all 600 saved fits. The portable rerun matched four tables and all 600 coefficient artifact hashes. The public package contains aggregates, fold assignments and hashes. It leaves individual supplied priors local.

```powershell
python task4/audit_local_pair_transfer.py --release-root <release-root> --output <new-output-directory>
```
