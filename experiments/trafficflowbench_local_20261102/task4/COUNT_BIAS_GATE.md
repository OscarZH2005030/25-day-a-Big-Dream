# Rejecting cross-input calibration on large positive count bias

The fixed rule rejects a 5% cross-input prior calibration when the loaded prior exceeds the sum of observed counts by more than 10% in any of the three inputs. Rejection returns the original baseline. The threshold was registered before generating the 160 artificial cases with seed 2026110329, covering ten networks, four demand families, four error mechanisms and moderate drift.

The rule admitted 78 cases. Relative to the baseline, the ungated calibration harmed 28 cases on the artificial Task4 composite and 32 on OD. The gated calibration harmed zero cases on either metric in this sample. Its mean composite gain fell from approximately 0.004016 to 0.003497. It rejected all 80 positive-background cases, including the independent-background cases where calibration had a positive mean gain.

On the three released inputs per network, the rule would admit only D12_I405_N and D12_I405_S. The second network has substantial negative loaded-prior bias, which illustrates the scope of a one-sided rule. Public activation is an input diagnostic. Hidden OD effects remain unknown.

Independent readback verified all 480 action scores and gate decisions, including exact fallback predictions. The portable runner reproduced the two experiment tables and all 320 local input/model/prediction artifact hashes. The artificial mechanism families were already studied, and this is a development result. A count-total rule cannot detect all matching errors. The official reference was retained.

```powershell
python task4/stress_count_bias_gate.py --release-root <release-root> --output <new-output-directory>
```
