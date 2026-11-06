# Faithful local Task 3 scorer (2026-11-05)

The public evaluator cannot score Task 3 locally: conservation is checked against organizer
boundary flows that are never published, so `S_LWR` floors at zero for everyone. On the train
split the observation layer is complete, so this scorer derives the boundary flows from the
truth (the truth then satisfies the conservation identity exactly) and otherwise calls the
official `score_task3.py` functions (`network_parameters`, `load_ramp_flows`, `link_order`,
`derive_physics`, `score_panel`). Only two things are substituted: the hybrid grid (truth plus
the submitted cells, read from a holdout window) and the boundary flux.

## Validation
- D12_I5_N, a time-of-day-mean control scores `S_physics` 0.3242, the same figure the
  official docs quote for their naive baseline.
- Truth as the prediction scores 0.9967 (the organizer's noisy flux gives 0.9598).
- Official flux carries noise, so absolute numbers here are about 0.03-0.04 higher than the
  leaderboard would show. Compare methods with each other, not with the leaderboard.

## Holdout results (train split, 2031-01-05..02-28, ten directions, mean of panels and regimes)
| method | S_FD | S_LWR | S_physics |
|---|---|---|---|
| A_dynamic_fixed | 0.988 | 0.605 | 0.733 |
| Current_fusion | 0.988 | 0.605 | 0.732 |
| T1 | 0.988 | 0.602 | 0.730 |
| LightGBM | 0.988 | 0.577 | 0.714 |

By regime (A_dynamic_fixed) S_LWR is 0.753 / 0.637 / 0.425 for R1 / R2 / R3. Full table:
`faithful_lwr_scores_10dir.csv`.

## Post-processing screen (D12_I5_N, Current_fusion base S_physics 0.7713, S_state proxy 0.9583)
All variants lost: moving-average smoothing (-0.004 to -0.081 S_physics), N-domain smoothing,
linear N interpolation blends (E_LWR rises with blend weight), ensembling with LightGBM or all
four methods (`ens_scores.csv`). Model N error (median 1.8%) is already below interpolation
error (2.5%); overall median true/model N is 0.999 (no global bias; a binned bias fix was not
completed). `diag_topo.py` shows topology flux from truth flows misses dN by E ~ 280, so a
conservation reconstruction from public neighbours is not possible.

## Files
- `faithful_lwr_scorer.py` the scorer (`--panel D12_I5_N --controls` reproduces the validation).
- `postproc_test.py`, `ens_test.py` post-processing and ensemble screens.
- `diag_N.py`, `diag_topo.py` N-error and topology-flux diagnostics.

Paths are hardcoded for the author's machine (`C:/Users/ROG/...`); edit the constants at the
top of `faithful_lwr_scorer.py`. Requires pandas, numpy, pyarrow and the official
`trafficflowbench-public` checkout.
