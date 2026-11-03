Local severe-mask mixture candidate. No Kaggle upload.
Parent is the frozen December/January residual candidate. New models use original50%/station15-augmentation25%/block30-augmentation25% public training weights.
January alone selects per-road gamma0/.25/.5/1, with original-state and per-condition accumulation-error guards.
Inference clips parent and augmented predictions independently, then mixes; this exact order is required. All features use public masked offline Task1 observations. Do not use future-observation features for Task2.
February development state and accumulation means improved, as did two new-seed severe artificial conditions. Individual roads can worsen. These are adaptively reused exposed public data, not official Task3 or leaderboard improvement.
Station15 hides eligible observable cells; noneligible signals may remain, so it is an outage sensitivity scenario rather than complete signal removal.
Preserve original official0.81135 baseline, all earlier candidates, and two remaining Kaggle opportunities. State files are not complete submissions; no Task2/Task4 merge or upload.
