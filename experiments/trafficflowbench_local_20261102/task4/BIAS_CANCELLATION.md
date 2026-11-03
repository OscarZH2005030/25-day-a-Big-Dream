# Bulk normalization hides positive background from the count-total gate

The paired stress test holds generated truth, counts and folds fixed while rescaling each supplied-style prior to match its own observed count total. It uses seed 2026110330, ten networks, four demand families, shared positive background and moderate drift. The original gate and 5% calibration remain fixed. Each variant is compared with its own baseline.

The gate rejected all 40 original cases and admitted all 40 normalized cases. Calibration reduced artificial OD accuracy in every admitted case, with a mean S_od change of -0.002439. Mean composite change was -0.000975 and the worst change was -0.003171. Bulk agreement can conceal pairing errors even when the link score barely changes. This mechanism was constructed to test the gate and does not describe verified public data generation.

Independent readback verified 240 action scores and paired inputs. The first portable comparison failed its byte-exact requirement. The 80 artifacts agreed numerically, with 16 byte-identical artifacts. Folds and seeds were exact, maximum normalized array difference was below 1e-14, and composite score differences were below 3e-16. Gate decisions and harm counts agreed. The reproduction record retains the exact comparison failure.

```powershell
python task4/stress_bias_cancellation.py --release-root <release-root> --output <new-output-directory>
```
