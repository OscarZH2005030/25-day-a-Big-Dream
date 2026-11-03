# Numerical conditioning of idealized OD inversion

Seed 2026110344 draws an integer OD flow on each released network. A known pulse and zone-index delays define origin-specific arrival kernels at each cut. The experiment broadens those kernels with a specified Gaussian spread and adds a signed perturbation with 1% relative L2 norm at each cut. It deconvolves each cut independently, then differences consecutive counts. Neither the pulse nor the perturbation model is fitted to traffic observations.

|Spread|Median OD relative L2, cutoff 1e-12|Median OD relative L2, cutoff 0.01|
|---:|---:|---:|
|0|0.241|0.241|
|1|5.026|5.026|
|2|2.118e6|4.187|
|4|1.731e10|3.112|

With wider arrivals the numerical inverse amplifies small measurement errors. The stronger singular-value cutoff reduces amplification and discards information. Even the zero-spread case accumulates error through differencing large downstream totals. These numbers describe this per-cut estimator, with negative reconstructed paths retained. They are neither lower bounds for other estimators nor measured detector performance. Singular-value ratios near machine precision are numerical diagnostics.

Independent readback checks all 80 error rows and verifies the 1% perturbations. An alternative SVD driver reproduces the 40 strongly truncated inverses. The portable run reproduces all result tables and 40 artifact hashes exactly.

```powershell
python task4/audit_hypothetical_time_conditioning.py --release-root <release-root> --output <new-output-directory>
```
