# OD ambiguity at the minimum invisible-L1 objective

Seed 2026110345 supplies one fixed linear direction per validation/private group. Opposite objectives choose two nonnegative flows at the same minimum invisible-L1 objective and with the same origin and destination totals as the saved optimum. The count predictions therefore also agree.

All twenty groups admit different OD allocations. The observed relative L1 separation is 0.0210 to 0.0698, with median 0.0398. The two directions are feasible examples, not a maximum L1 separation bound. Distance to the original supplied prior can change even though distance to the signed repair stays fixed.

Independent readback confirms both gross margins, count predictions and objective equality for all forty flows. The portable run rebuilds the earlier net and L1 optima and matches the tables and twenty artifact hashes. These are diagnostics of tie-breaking ambiguity. They provide no hidden OD score and the alternatives remain local.

```powershell
python task4/audit_minimum_repair_face.py --release-root <release-root> --output <new-output-directory>
```
