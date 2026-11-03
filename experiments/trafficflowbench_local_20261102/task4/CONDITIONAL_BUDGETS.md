# Conditional movement budgets and demand lower bounds

Preserving count fit, origin/destination margins and original-prior L1 distance fixes three Task4 components. It does not fix OD accuracy. If the unknown nonnegative reference satisfies `||A f* - c||1 <= epsilon sum(c)`, any vector `y` with `A.T y <= 1` gives

`sum(f*) >= c.T y - epsilon sum(c) ||y||infinity`.

This follows from `sum(f*) >= y.T A f*` and the L1/L-infinity inequality. Maximizing this expression is a linear program. The simple bound `(1-epsilon) sum(c) / max_column_sum(A)` is also retained as a fallback. A candidate in the same feasible orthant segment with L1 movement at most `rho * lower_bound` then has absolute S_od change at most rho. If the other three components remain equal, its absolute Task4 change is at most `0.45 rho` (and total change at most `0.09 rho` if the Task4 weight is 20%).

The hidden-reference compatibility assumption is unverified. This is not an unconditional safety guarantee, confidence interval, improvement prediction or reason to replace the official reference. A tighter lower bound only permits more movement at the same conditional radius; it cannot make the movement direction correct.

For all ten released networks and three public input groups, the optimized lower bound improves capacity relative to the simple bound as follows:

|Assumed count NMAE|Minimum ratio|Mean ratio|Maximum ratio|
|---|---:|---:|---:|
|0|1.2183|3.3140|4.7898|
|0.01|1.0303|2.1825|3.5914|
|0.05|1.0000|1.0257|1.1483|

All 90 dual solutions were rescaled for numerical feasibility and checked against separately constructed original-variable primal LPs. Maximum relative primal-dual gap was approximately 1.0002e-10. These are floating-point numerical checks, not formal exact-arithmetic certificates. Public input groups are already exposed and contain no hidden-reference validation labels.

Separately, 3840 movement-budget actions on the existing 480 artificial cases passed independent readback. That is a numerical sensitivity check on reused cases, not fresh empirical validation. A compatible four-cell example reaches the `0.45 rho` limit exactly. With the same observables and two possible matching references, the preferred direction reverses. A count-incompatible example has true NMAE 0.99 and OD loss 0.1 despite an assumed-epsilon-zero movement budget with rho 0.001, showing why the compatibility condition cannot be omitted.

Portable count-geometry and independent primal reproduction:

```powershell
python task4/audit_budget_bounds.py --release-root <release-root> --output <new-output-directory>
```

No real replacement flow or Kaggle upload was produced. Original data, all candidates and the official tested Task4 baseline remain preserved.
