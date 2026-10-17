# 小组成员：查看结果与换电脑复现

只看结果无需下载数据、安装依赖或重跑：直接打开本目录 README 和 `reports/summary.csv`。

下面在 Windows PowerShell 中操作。先克隆小组仓库，再进入 `experiments/task1_january_grin`。本目录相当于原电脑上的“IEEE比赛”项目根目录，所有命令都从这里运行。旧说明中的 `C:\Users\DELL\...` 不需要照抄。

## 1. 数据和环境

将自己下载的官方发布数据放成：

```text
task1_january_grin/
  task1/
  2026-ieee-big-data-traffic-flow-bench/
    kaggle_public/
      README.md
      config/
      corridors/
      task1/
      ...
```

使用 Python 3.14（原实验为 3.14.8），先检查 `python --version`。安装原锁定依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_task1.ps1
powershell -ExecutionPolicy Bypass -File .\setup_grin.ps1 -TorchBuild cu128
.\.venv-grin\Scripts\python.exe -m pytest tests -q
```

GPU 复现使用 PyTorch 2.10.0+cu128，RTX 3060 的 batch size 使用 2。CPU 可把安装参数换成 `-TorchBuild cpu`、训练参数换成 `--device cpu`，速度会不同。安装时若旧版包镜像没有锁定版本，应先核对环境，不要默默升级依赖后声称完全复现。

## 2. 纯 LightGBM baseline

若要重跑十方向冻结 baseline（包括残差、直接、等权和 60 分钟规则）：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_all_baseline.ps1 -Mode run -Threads 4
```

最初 B0 历史均值、B1 插值、B2 残差 LightGBM 的三方法实验仍可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode dev -Threads 4
```

接下来为 GRIN 两方向对照生成当前电脑上的完整 baseline 预测和缓存（可以复用本地已有同配置缓存）。输出名固定写出来，便于后续命令衔接：

```powershell
.\.venv\Scripts\python.exe -m task1 corridor-validate --panels D12_I5_N D7_I405_N --output runs/team_baseline --threads 4
.\.venv\Scripts\python.exe prepare_team_references.py --baseline-run runs/team_baseline --output runs/team_reference
```

辅助脚本仅从已完成、完整两方向的 baseline 复制评分参考并校验协议，不训练模型、不改写原结果。它为现有 GRIN 入口提供本机可用的 reference，避免依赖原电脑的绝对路径。

## 3. I5 GRIN 和组合

明确取本机新 baseline 的训练目标表，以核对抽样完全一致：

```powershell
$baselineCfg = Get-Content -LiteralPath .\runs\team_baseline\run_config.json -Raw | ConvertFrom-Json
$usage = Get-Content -LiteralPath .\runs\team_baseline\cache_usage.json -Raw | ConvertFrom-Json
$i5Usage = $usage | Where-Object panel -EQ 'D12_I5_N'
$i5Sample = Join-Path $baselineCfg.cache_dir (Join-Path $i5Usage.training_key 'training.parquet')
.\.venv-grin\Scripts\python.exe -m task1.grin run --output runs/team_grin_i5 --baseline-run runs/team_reference --baseline-sample-table $i5Sample --panel D12_I5_N --device cuda --batch-size 2
```

保持默认 hidden=32、window=48、stride=24、20 epochs、每轮100步、lr=0.001。完成后只复用预测计算 I5 组合：

```powershell
.\.venv-grin\Scripts\python.exe -m task1.grin.hybrid combine --panel D12_I5_N --baseline-run runs/team_baseline --frozen-reference runs/team_reference --template-grin-run runs/team_grin_i5 --grin-run runs/team_grin_i5 --output runs/team_hybrid_i5 --device cuda --batch-size 2
```

## 4. D7 验证

使用刚刚完成的 I5 配置作为模板，在 D7 自身训练期独立训练；阈值固定为 60 分钟：

```powershell
.\.venv-grin\Scripts\python.exe -m task1.grin.hybrid validate --panel D7_I405_N --baseline-run runs/team_baseline --frozen-reference runs/team_reference --template-grin-run runs/team_grin_i5 --output runs/team_hybrid_d7 --device cuda --batch-size 2
```

新电脑首次会训练 D7 GRIN；存在同配置、通过完整性检查的预测时会复用。正式成功后会自动追加本目录的 `实验日志.txt`。检查 `comparison_hybrid.csv`、`metrics_by_panel_regime.csv`、各 `*_gap_errors.csv` / `*_daily_errors.csv` / `*_congestion_errors.csv` 及 `log_update.json`。

已有输出不会被覆盖。需要重复实验时换新输出名，同时修改后续引用；需要变更超参数时单独记录实验，不修改本次历史结果。GPU/依赖版本及数值运算可能使新训练结果略有差异，先核对样本一致性和 baseline 复现检查。

## 5. 协作时提交什么

提交源代码、说明、新正式实验的小型结果表与实验日志。`runs/`、`cache/`、原始数据、环境、权重和逐目标预测均已忽略。本次历史报告在 `reports/` 中，不要把它直接复制成 `runs/` 后当作完整模型运行。

若小组需要免训练共享权重和逐目标预测，可另打模型/预测包共享，保持自然键、配置及缓存清单一起提供。本次 Git 快照不包含这些运行产物。
