# Road-panel sensitivity of the Task4 amplitude comparison

This descriptive check pairs full and quarter amplitude results within each of 80 reused synthetic cases. It averages within ten road panels, omits each panel in turn, and draws 10,000 whole-panel resamples using seed 2026110353. The same draw indices are used for all eight noise-rule combinations.

All eight central 95% resample ranges span zero. For the agreement rule, either background-noise mean can change sign after a single panel is dropped. The two lognormal means stay negative under every single-panel omission. The intervals describe the sensitivity of these artificial cases to panel composition. They do not estimate performance on unseen roads or hidden OD targets.

After running the quarter-amplitude entry point, use the same output directory.

    python task4/stress_source_error_amplitude_panels.py --output /path/to/output
