# Fixed source clipping across four error mechanisms

Seed 2026110338 extends the fixed 95th-percentile source-residual clipping rule to 160 artificial cases. It covers ten networks, four demand families, four error mechanisms and moderate drift. Count-shape TV is fixed at 0.14. The forecast code, fold-local scaling and 5% blend come from the preceding 40-case study.

|Artificial mechanism|Raw mean Task4 change|Clipped mean Task4 change|Raw to clipped composite harms / 40|
|---|---:|---:|---:|
|Independent lognormal|+0.007273|+0.008575|0 to 2|
|Independent positive background|+0.001068|+0.001869|3 to 1|
|Shared lognormal|+0.002002|+0.003471|0 to 3|
|Shared positive background|-0.000140|+0.000603|25 to 16|

The clipped forecast improves the mean in every mechanism in this stress. It adds composite harms in the two lognormal families even though both forecasts improve OD in all of those cases. Under shared positive background, clipping improves mean OD change from -0.000655 to +0.000934 and reduces OD harms from 29 to 19. This is a tradeoff between mean effects and individual cases. The artificial study uses exposed networks and leaves hidden OD accuracy unverified.

Independent readback checks the 480 actions and 1,600 fold fits. The full portable runner reproduces the tables and 160 artifact hashes. All registered cases completed.

```powershell
python task4/stress_source_clip_extended.py --release-root <release-root> --output <new-output-directory>
```
