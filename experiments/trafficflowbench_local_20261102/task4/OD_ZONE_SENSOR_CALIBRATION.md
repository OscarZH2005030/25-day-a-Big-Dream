# Calibrating shared error in hypothetical origin-total measurements

The experiment freezes four sets of 16 origin-total measurements across ten roads, three prior variance laws, and two independent measurement noise levels. It adds a common zero-mean offset with variance equal to the independent noise, then conditions on one independent reference measurement of the offset. Reference error variance is 0.25, 1, or 4 times the offset variance. Each case checks the rank-one update against a joint covariance solve with the reference included. No random sampling is used.

At the best tested reference precision the calibration recovered 53.86% and 34.48% of the mean gain lost to bias across the two noise levels. At the weakest tested precision it recovered 6.82% and 3.19%. These are model-based origin-total variance reductions for an assumed additive bias and a reference that measures that bias directly. They provide no hidden OD validation or evidence that a matching reference exists in the public road measurements.

Run from the experiment directory with the public release available locally.

    python task4/audit_od_zone_sensor_calibration.py --release-root /path/to/release --output /path/to/output
