# Local path caps at equal total OD movement

The follow-up reuses the forty budgeted source cases and their clipped source objective. It limits each path change to 5% of baseline flow, in addition to the 0.002 global L1 cap. The global-only direction is scaled to the realized movement of the locally capped solution. Counts, margins and original-prior L1 remain fixed. Both solutions use the full global cap in these cases.

|Artificial mechanism|Method|Mean moved paths|Mean OD change|OD harms / 20|
|---|---:|---:|---:|---:|
|Shared lognormal|Global cap|20.10|-0.000526|10|
|Shared lognormal|Local and global caps|140.70|-0.000131|9|
|Shared positive background|Global cap|32.75|+0.000184|6|
|Shared positive background|Local and global caps|167.40|+0.000620|3|

The local cap spreads the same amount of movement over more paths and improves both artificial means relative to the global-only direction. The lognormal mean remains below baseline. The cap was tested on reused cases and does not establish hidden improvement.

Independent readback verifies eighty actions, the per-path and global bounds, matched movement and non-OD invariants. The portable chain reproduces the result files and all forty case hashes.

```powershell
python task4/audit_budgeted_source_path_cap.py --release-root <release-root> --output <new-output-directory>
```
