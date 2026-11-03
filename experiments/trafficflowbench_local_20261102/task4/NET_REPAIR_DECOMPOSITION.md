# Count-visible and invisible components of the frozen correction

All twenty validation/private groups map their released count rows to the complete set of zone cuts. Repeated rows agree within numerical tolerance. Consecutive cut differences recover the zone net vector under the static exact-count interpretation.

On the complete triangular support, the zone incidence matrix satisfies the complete-graph Laplacian identity. For zero-sum net imbalance `r`, the signed minimum-L2 repair on path `(o,d)` is `(r[o]-r[d])/n`. Adding this repair to the supplied prior matches released counts, but 23.3% to 31.1% of paths become negative across the twenty groups.

Projecting the frozen prediction's change from the prior onto the count row space leaves a count-null component. Its share of squared correction norm ranges from 9.5% to 53.5%, with a median of 39.2%. This measures how much of the existing correction lies outside the count-visible subspace. It does not classify that movement as accurate or unnecessary. Nonnegativity can require count-null movement.

Independent count-space SVD verifies the signed repair and both components. The portable audit uses the frozen prediction CSV and reproduces the tables and all twenty artifact hashes. The diagnostic does not export alternative path flows or infer hidden accuracy.

```powershell
python task4/audit_task4_net_repair.py --release-root <release-root> --output <new-output-directory>
```
