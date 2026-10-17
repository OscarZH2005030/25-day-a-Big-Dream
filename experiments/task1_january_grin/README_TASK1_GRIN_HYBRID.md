# baseline＋GRIN 固定缺口切换

## 运行

沿用现有 `.venv-grin`，无需新增依赖。正式验证 D7_I405_N：

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'
powershell -ExecutionPolicy Bypass -File .\run_task1_grin_hybrid.ps1 -Mode d7 -Device cuda -BatchSize 2
```

这条命令自动完成：复用 D7 已有冻结 baseline 预测 → 查找配置匹配的已完成 GRIN → 没有时在 D7 自身训练期训练 GRIN → 比较 baseline、GRIN、组合 → 更新日志。没有 LightGBM 训练步骤。

I5 仅复用已有预测，复现组合分数，不训练、不做神经网络推理：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_grin_hybrid.ps1 -Mode i5
```

## 固定规则与公平比较

- 双侧可见**速度**观测间隔有限且 **>60 分钟：使用 GRIN**。
- 其余：使用冻结 baseline；恰好 60 分钟、缺任一侧、两侧都缺时均使用 baseline。
- 两个输出共用同一选择条件。冻结 baseline 自身仍为原来的多跳残差/直接预测＋60 分钟规则。
- 按完整 panel/timestamp/station/link/regime 键对齐，不依赖两个文件行顺序；不使用真值或真实拥堵状态选择模型。
- 已保存流量为整条 link 的 veh/h；切换不再次乘车道数。评分继续复用 `task1.metrics` 的每车道归一化及 regime → 方向 → 走廊等权聚合。

该规则来源于 I5 一月开发分析，本轮在 D7 前固定，不扫阈值、不调融合权重。

D7 的 GRIN 从头在其自身 2030-06 至 12 月训练，完整一月评测；二月保持留出。不是把 I5 模型直接迁移到另一走廊。默认沿用第一轮 I5 GRIN 配置：hidden=32、window=48、stride=24、batch=2、lr=0.001、最多 20 epochs、每轮最多 100 步、patience=5、随机遮挡 5%、seed=20261016、原缺失监督每天 1000 个目标。训练样本键会与 D7 baseline 缓存核对。

GRIN 仍按其**单模型**的一月 S_state 选择最佳 checkpoint，再使用固定切换规则评测；不根据组合分数重新选择 checkpoint。学习率和模型结构优化留作后续独立实验。

## 复用与校验

- I5 baseline：复用原多跳残差/直接预测，必要时重建原冻结融合；GRIN 复用已有正式预测。
- D7 baseline：直接读取 `corridor_dev_20261016_192125` 的 `Fusion_gap60mean_predictions.parquet`；从原预测缓存读取可见观测间隔。
- GRIN 图张量继续使用 `cache\grin`；没有改动张量构建文件或缓存身份，因此原 I5 缓存保持有效。D7 首次需要建立自己的图张量缓存。
- 自动查找已完成的独立 GRIN 结果和本流程的 `grin_run` 子目录。仅当方向、数据期、抽样、训练配置、模型/数据代码与评分一致时复用；复用预测后不再训练或推理。
- 未完成的训练目录不会自动当作已完成结果。再次运行会复用已发布的图张量缓存；若没有匹配的已完成 GRIN，会重新开始训练。
- baseline 和 GRIN 的分数都重新按 regime 核对；缓存内容、可见原始输入做哈希校验，目标必须完整匹配；不匹配则报错。

可显式指定已有 GRIN 结果：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_grin_hybrid.ps1 -Mode d7 -GrinRun '.\runs\已完成的D7结果\grin_run'
```

训练配置不同的结果不会被默默复用。`-BatchSize` 改变时属于不同训练配置；默认 2 与 I5 第一轮一致。`-BaselineRun` 可指定兼容的原 target/corridor 正式结果，默认路径已配置好。

## 输出与日志

输出目录为 `runs\grin_hybrid_d7_时间戳` 或 `grin_hybrid_i5_时间戳`：

- **comparison_hybrid.csv**：baseline、GRIN、组合的总分、速度/每车道流量 RMSE，及相对 baseline、GRIN 的增益。
- `metrics_by_panel_regime.csv`、`metrics_summary.csv`：R1/R2/R3 与聚合分数。
- `*_daily_errors.csv`、`*_gap_errors.csv`、`*_congestion_errors.csv`：三种方法的分日、缺口、拥堵误差。
- `combined_predictions.parquet`：组合预测，附带缺口分组和实际选择的模型；不复制两个原模型的预测文件。
- `gate_counts.csv`：各 regime/缺口组选择 GRIN 或 baseline 的目标数。
- `reuse.json`：baseline 是否复用、GRIN 预测来源、本次是否训练 GRIN、图缓存状态。
- `reference_check.json`、`source_manifest.json`：逐 regime 复现检查、输入文件哈希。
- `grin_run\`：仅本次需要训练 GRIN 时创建；包含 `best.pt`、训练曲线、抽样检查、模型配置及 GRIN 原始预测。
- `COMPLETE.json`、`log_update.json`：完成标记与日志状态。

GRIN 子训练使用 `no_log`，由外层组合流程统一把三种方法、分 regime 得分、切换数、训练配置和复用状态追加到根目录 **实验日志.txt**；保留历史并按结果目录去重。正式成功应显示 `Log status: appended`。再次运行相同配置会生成新的结果目录，可用于验证复用。

如果日志追加失败，结果保留，执行补记，无需重新训练：

```powershell
.\.venv-grin\Scripts\python.exe -m task1.grin.hybrid log --run '.\runs\grin_hybrid_d7_实际时间戳'
```

小规模检查使用 D7 前 7 天训练、前 3 天评测，训练预算与 GRIN smoke 一致；baseline 仍复用原正式模型的这三天预测。因此它仅检查流程，不作正式模型对比、不写正式日志：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_grin_hybrid.ps1 -Mode smoke -Device cuda -BatchSize 2
.\.venv-grin\Scripts\python.exe -m pytest tests -q
```

`-NoLog` 可显式跳过日志，`-Output` 可指定项目内新的空结果目录。首次 D7 运行会训练 GRIN；I5 模式只组合现有结果。验证后应同时查看缺口分项，不能仅依据某个走廊的总分断言普遍提升。
