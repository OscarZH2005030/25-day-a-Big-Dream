# Task 1 and Task 2 baseline results

As of 2026-10-17. Task 1 is shared by the whole team; Oscar owns Task 2. These
are local development results, not Kaggle leaderboard scores. The released
score weights are 35% Task 1, 30% Task 2, 15% Task 3 and 20% Task 4. Task 3
reads the Task 1 state predictions, so state reconstruction also affects the
physics score. See the [competition evaluation page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/evaluation).

## Task 1: weekday/time historical mean

### Method

For every detector link, calculate a mean speed and flow for each weekday and
five-minute time-of-day slot. Use eligible, finite observations from the first
eight months of `train` (2030-06-01 through 2031-01-31). For each official Task 1
target row in the last train month (2031-02-01 through 2031-02-28), look up the
matching link/weekday/time profile. Use the link-wide training mean if that
specific slot has no training observations. Flow is evaluated per lane, as in
the competition metric.

This time split keeps the target month out of profile fitting. It is a more
useful local estimate than fitting and scoring on the same train rows.

### Result

The local `S_state` is **0.730602** across the five corridor families. The
calculation averages R1/R2/R3 within each direction, averages the two
directions within each corridor family, then gives the five families equal
weight. It evaluated **3,443,944** official Task 1 target rows from the final
train month.

| Corridor family | Time-holdout `S_state` |
| --- | ---: |
| D12_I405 | 0.8130 |
| D12_I5 | 0.6466 |
| D7_I10 | 0.7657 |
| D7_I210 | 0.8439 |
| D7_I405 | 0.5837 |
| **Equal-family mean** | **0.7306** |

| Mask regime | Mean `S_state` | Mean speed score | Mean flow score |
| --- | ---: | ---: | ---: |
| R1 | 0.7471 | 0.6500 | 0.8611 |
| R2 | 0.7280 | 0.6320 | 0.8408 |
| R3 | 0.7166 | 0.6138 | 0.8374 |

The flow score is stronger than the speed score in every regime, and the 50%
mask regime is hardest. `D7_I405` and `D12_I5` are the lowest-scoring families.
That points to the limits of a link's average pattern when traffic deviates
from its normal weekday/time profile, especially around congestion and
incidents. It does not establish that those corridors are intrinsically harder;
the result is specific to this one holdout month.

### Prediction files and checks

The same method was fit on all eligible train observations and used to create
candidate baseline files for the actual hidden splits:

| Split | Rows | Check |
| --- | ---: | --- |
| validation | 3,451,376 | Exact match to the released Task 1 template keys; all values finite and nonnegative |
| private | 3,284,419 | Exact match to the released Task 1 template keys; all values finite and nonnegative |

The generated CSVs are in the ignored local `outputs/` folder so the 11.5 GiB
dataset and large predictions are not added to Git. Regenerate them with:

```powershell
python scripts/build_task1_historical_mean.py --split both
```

### Limitations

- The baseline does not use the current day's visible observations, adjacent
  links, upstream/downstream queue movement, or traffic conservation.
- Speed is the main weakness: the mean speed score was 0.61–0.65 while mean
  per-lane flow score was 0.84–0.86.
- The result is a train-month holdout score. It is not the public leaderboard
  score and cannot predict performance on the independent private month.
- The upstream convenience evaluator `baseline_task1_historical_mean.py` was
  also run against the supplied reissued data. It reported empty regime cells
  for some corridor/split combinations, assigning those empty groups a score of
  1.0. Those values are not used here. The time-holdout result above uses
  complete unmasked train answers and the official train target template.

## Task 2: persistence baseline

### Method

For each queue window, read the last released history timestamp (`T-5`). Mark a
link as queued if its observed speed is at or below 60% of its free-flow speed,
then repeat that binary state for all six requested future times (`T+5` through
`T+30`). The upstream baseline deliberately does not use the released row at
`T`, even though the rules allow it.

### Result

The training proxy uses complete unmasked train speeds to derive queue labels
with the published threshold and applies the space-time IoU over eligible
link/time cells. It covers **80 windows**: 40 `queue_onset` and 40
`queue_ongoing` across the eight scored panels. The released train speed is a
measured value, while the official queue label comes from the simulator's
underlying state. Measurement noise can change labels near the threshold, so
this is a diagnostic proxy rather than an exact reproduction of the official
queue score.

| Train window condition | Windows | Mean IoU |
| --- | ---: | ---: |
| queue_onset | 40 | 0.0250 |
| queue_ongoing | 40 | 0.5898 |
| **All train windows** | **80** | **0.3074** |

The method recognizes queues that are already visible, but contributes almost
nothing when a queue first forms. Persistence predicts zero across all onset
windows by construction; small nonzero IoU can result from noisy observations
near the threshold. This is the clearest improvement target for Task 2.

The upstream repository reports **0.3017** for its validation persistence
baseline. That is a reference score from the competition's validation split;
it is separate from our 0.3074 train proxy, and should not be compared as if
they were the same evaluation sample. The released package has no queue-label
file, so the local proxy derives labels from the complete train speeds.

The baseline predictions for train, validation and private each contain
**87,000 rows**. Their keys match the released templates after UTC timestamp
normalization, and every `queue_pred` is binary. Regenerate with:

```powershell
python vendor/trafficflowbench-public/src/task2/build_task2_persistence_submission.py `
  --release-root data/kaggle_public --split train `
  --output outputs/task2_persistence_train.csv
python vendor/trafficflowbench-public/src/task2/build_task2_persistence_submission.py `
  --release-root data/kaggle_public --split validation `
  --output outputs/task2_persistence_validation.csv
python vendor/trafficflowbench-public/src/task2/build_task2_persistence_submission.py `
  --release-root data/kaggle_public --split private `
  --output outputs/task2_persistence_private.csv
python scripts/evaluate_task2_train_proxy.py
```

### Limitations

- The baseline assumes every queue stays in the same place and at the same
  extent for 30 minutes. It cannot model queue growth, dissipation, or movement.
- Onset performance is nearly zero. A first improved model should separate
  `queue_onset` from `queue_ongoing` or learn a per-link, per-horizon queue
  probability from the preceding 60-minute history.
- The local train proxy is useful for iteration but is not a substitute for
  validation leaderboard scoring. Validation and private have different
  incidents and no released queue labels. It also derives labels from measured
  speed, which can differ from the hidden underlying-state queue labels near
  the cutoff.
- Task 2 excludes `D12_I405_N` and `D12_I405_S`; those panels remain part of
  Task 1 and Task 3.

## Reproduction and team handoff

Run all baseline checks with:

```powershell
python scripts/validate_oscar_baselines.py
```

Task 1 and Task 2 outputs remain per-task files. They are not the final Kaggle
CSV; the team integration owner must merge them with Task 4 using the official
`submission_key.csv` and submission template.

The team plan assigns Task 1+3 to role A, Task 2 to role B, and Task 4 plus
engineering integration to role C. The team clarified that Task 1 is shared
and Oscar is role B, owning Task 2. The screenshots do not map A/C to names.
