# Compare directions at equal movement

The minimum-movement tie rule changes both the magnitude and direction of the baseline correction. To isolate direction, this follow-up scales the default direction to the same L1 length as the minimum-movement direction on each of the forty reused artificial cases. Both paths stay on the same non-OD score face. The scaled default endpoint need not attain the primary LP optimum.

|Mechanism|Fraction of matched movement|Minimum-move mean OD change|Scaled-default mean OD change|Minimum-move wins / losses / ties|
|---|---:|---:|---:|---:|
|Shared lognormal|5%|+0.000056|+0.000192|0 / 14 / 6|
|Shared lognormal|100%|-0.000477|+0.003102|0 / 14 / 6|
|Shared positive background|5%|-0.000030|-0.000023|6 / 12 / 2|
|Shared positive background|100%|-0.001743|-0.001118|4 / 14 / 2|

The apparent reduction in full-endpoint losses under minimum-movement tie-breaking does not carry over to a matched-movement advantage in these cases. Reusing the cases makes this a descriptive comparison. It does not select a deployment direction.

The first verification attempt stopped at a malformed context-manager continuation. Correcting the verifier left the experiment untouched. Independent readback checks 160 actions, movement equality and score invariants. The full portable chain reproduces the three result tables exactly.

```powershell
python task4/audit_equal_movement_ties.py --release-root <release-root> --output <new-output-directory>
```
