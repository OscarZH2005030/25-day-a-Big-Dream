# Minimum count-invisible repair under nonnegativity

The signed minimum-L2 repair can match a fixed net vector while producing negative paths. This diagnostic holds that net vector fixed and minimizes the L1 movement from the signed flow needed to restore nonnegativity. It evaluates released net vectors and the frozen prediction's net vectors separately.

For the frozen net vectors, the optimum invisible L1 is 76.3% to 84.7% of the frozen prediction's invisible L1, with median 80.3%. Both flows have identical count predictions in this comparison. The minimum still requires substantial invisible movement. It is a minimum for this particular L1 objective, not an OD accuracy or scoring optimum.

Independent checks verify primal feasibility, reduced costs and dual gaps on forty LPs. The complete portable run matches the four result files and all forty artifacts. Alternative path vectors stay local. Released net targets are a separate conditional analysis, since compatibility with hidden reference counts remains unverified.

```powershell
python task4/audit_minimum_invisible_repair.py --release-root <release-root> --output <new-output-directory>
```
