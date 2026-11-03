# Source clipping when the target pairing is redrawn

Seed 2026110339 repeats the fixed source clipping comparison with independently redrawn latent pairing in each input. The experiment covers ten networks, four demand families, four error mechanisms and 160 artificial cases at TV 0.14. Both methods use the same fixed 5% correction. The moderate-drift experiment used a different seed, so the comparison between studies also reflects different draws.

|Artificial mechanism|Raw mean Task4 difference|Clipped mean difference|Raw to clipped harms / 40|
|---|---:|---:|---:|
|Independent lognormal|+0.006946|+0.006895|0 to 0|
|Independent positive background|-0.002244|-0.002242|21 to 20|
|Shared lognormal|+0.004899|+0.004427|1 to 1|
|Shared positive background|-0.003620|-0.003706|35 to 35|

The broad mean advantage seen under moderate drift does not recur in this redraw stress. The shared-background setting remains particularly adverse. The artificial draws offer no evidence about which drift or error mechanism describes hidden OD.

The independent check covers 480 action scores and 1,600 fold fits. The complete portable rerun compares the tables, failure list and all 160 local case hashes.

```powershell
python task4/stress_source_clip_replaced.py --release-root <release-root> --output <new-output-directory>
```
