# Minimum-movement tie-breaking on the matched-score face

The secondary LP keeps the minimum invisible-L1 value and the baseline's counts, margins and original-prior L1 distance fixed. Among primary optima it minimizes L1 movement from the baseline. It reuses the forty artificial cases from the matched-score shrinkage experiment.

|Artificial mechanism|Tie rule|Full endpoint mean relative L1 movement|Full endpoint mean OD change|
|---|---|---:|---:|
|Shared lognormal|Default|0.20969|-0.002823|
|Shared lognormal|Minimum movement|0.04251|-0.000477|
|Shared positive background|Default|0.09080|-0.009606|
|Shared positive background|Minimum movement|0.02446|-0.001743|

The secondary rule reduces the full endpoint's movement and average losses on these reused cases. At 5% interpolation it also reduces the positive lognormal mean from 0.001020 to 0.000056. A fixed interpolation fraction compares different amounts of movement. The results alone cannot establish a better direction or hidden OD accuracy.

Independent readback checks 160 actions, primary objective equality and invariant non-OD components. The portable chain rebuilds the cases and reproduces all three tables/failure files and forty artifact hashes.

```powershell
python task4/stress_invariant_shrink_tiebreak.py --release-root <release-root> --output <new-output-directory>
```
