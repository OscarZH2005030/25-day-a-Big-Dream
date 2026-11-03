# Fresh four-mechanism stress for the fixed local cap

Seed 2026110348 covers eighty new artificial cases with the previous fixed source clipping, 5% local path cap and 0.002 global L1 cap. Both methods use the same realized movement and preserve all three non-OD score components. The comparison holds the clipping rule fixed while changing how movement can be distributed over paths.

|Artificial mechanism|Matched global mean OD change|Local cap mean OD change|Harms global to local / 20|
|---|---:|---:|---:|
|Independent lognormal|+0.000184|+0.000574|8 to 3|
|Independent positive background|+0.000160|+0.000568|7 to 0|
|Shared lognormal|-0.000344|-0.000092|9 to 10|
|Shared positive background|-0.000062|+0.000443|8 to 5|

The local cap improves all four means relative to the global-only direction in this artificial stress. Shared lognormal still has a negative mean against baseline, and one more harmful case. The experiment does not establish that these mechanisms describe hidden OD errors.

Independent readback checks 240 scores and all matched movement, local cap and non-OD invariants. Portable reproduction matches all four result files and eighty local case hashes.

```powershell
python task4/stress_path_cap_extended.py --release-root <release-root> --output <new-output-directory>
```
