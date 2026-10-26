# 25-day-a-Big-Dream

Team workspace for the 2026 IEEE Big Data-Traffic Flow Bench Kaggle competition.
The competition has four scored tasks: traffic-state reconstruction, queue
forecasting, physics consistency, and OD/path-flow estimation.

## Current ownership

Task 1 (state reconstruction) is shared by the whole team. Oscar is role B and
owns Task 2 (queue prediction). The team plan assigns Task 1+3 to role A and
Task 4 plus engineering integration to role C; the screenshots do not map A/C
to names.

## Local data and environment

Download and extract the Kaggle data package so the release root is
`data/kaggle_public/`. The dataset is excluded from Git because it is about
11.5 GiB unpacked. Install Python requirements with:

```powershell
python -m pip install -r requirements.txt
```

The result files go into `outputs/`, which is also excluded from Git. The
per-task predictors and evaluation scripts are kept in `scripts/`; unmodified
official Task 1 and Task 2 baseline code is kept in
`vendor/trafficflowbench-public/` with its MIT license and attribution.

## Task 1 and Task 2 baselines

See [docs/OSCAR_TASK1_TASK2_BASELINE.md](docs/OSCAR_TASK1_TASK2_BASELINE.md) for
the methods, local scores, interpretation, caveats and commands used to
reproduce the baseline results.
中文复跑总结见 [docs/中文基线复跑总结.md](docs/中文基线复跑总结.md)。

Task 1 uses a time-based holdout: train the weekday-by-five-minute historical
profile before 2031-02-01, then score the official Task 1 target rows from the
last train month. Task 2 uses the official persistence baseline and derives
training queue labels from the complete unmasked train speeds for a local IoU
proxy.

## Useful commands

```powershell
python scripts/evaluate_task1_time_holdout.py
python scripts/build_task1_historical_mean.py --split both
python vendor/trafficflowbench-public/src/task2/build_task2_persistence_submission.py `
  --release-root data/kaggle_public --split train `
  --output outputs/task2_persistence_train.csv
python scripts/evaluate_task2_train_proxy.py
python scripts/validate_oscar_baselines.py
```

Task 2 validation and private predictions use the same builder with
`--split validation` or `--split private`. The Task 2 train IoU is a diagnostic
proxy, not a Kaggle leaderboard score. Public/private leaderboard scores are
available only after submitting on Kaggle.

## Task 1 learned baselines and gated residual (local experiments)

See [Task 1 experiment report](docs/TASK1_GATED_RESIDUAL.md) for the shared-51-feature
LightGBM/graph Transformer comparison, topology-only fallback, per-channel diagnosis,
and the in-progress cross-fitted gated residual. [Reproduction commands](experiments/task1/README.md)
use isolated output directories. These use a different local split from Oscar's
historical-mean baseline; the scores are not directly comparable. No new Kaggle
submission or Task 3 physics improvement is claimed.

## 2026-10-25：十道路逐路选优

[模型下载、中文说明与PowerShell运行指令](experiments/task1/road_selection_common55_20261025/README.md)。共同55天选模分数0.940519388；五路采用原GitHub组合，五路采用186特征适配LightGBM。
