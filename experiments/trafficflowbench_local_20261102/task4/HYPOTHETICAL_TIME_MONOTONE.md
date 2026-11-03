# Physical projections of the idealized inverse

This follow-up reuses the 80 hypothetical inverse cases and supplies their exact origin totals. It compares unmodified deconvolution, nonnegative path clipping with per-origin rescaling, and decreasing cumulative-flow projection with fixed origin totals and zero terminal flow. Both projections enforce nonnegative paths and the same origin totals.

For arrival spread 4 and the 0.01 spectral cutoff, the median OD relative L2 error is 3.112 before repair, 1.179 after clipping and scaling, and 1.248 after the monotone projection. Median observation relative L2 error is 0.00959, 0.15032 and 0.01426 respectively. The monotone projection fits the idealized observations better than clipping and scaling while having higher median OD error on these reused cases.

Independent checks verify 240 OD scores and the bounded monotone projection's first-order condition at every step-function vertex. The portable runner rebuilds the underlying inverse cases and matches the result tables exactly. The comparison assumes hypothetical exact origin totals and does not validate a real detector, timing model or official candidate.

```powershell
python task4/audit_hypothetical_time_monotone.py --release-root <release-root> --output <new-output-directory>
```
