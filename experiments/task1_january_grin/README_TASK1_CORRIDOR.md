# Task 1：另一条走廊验证

正式运行：

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'
powershell -ExecutionPolicy Bypass -File .\run_task1_corridor.ps1 -Mode dev -Threads 4
```

默认方向 **D7_I405_N**，属于与原 D12_I5_N 不同的走廊家族。训练用新走廊自身 2030-06-01 至 2030-12-31 的数据；评测为完整 2031 年 1 月。固定每天抽样 1000、350 轮、seed=20261016，使用原多跳特征和 LightGBM 参数；两个目标模式共享样本与特征。验证的是方案在另一走廊独立训练后是否重复有效。不是把 I5 模型直接迁移过去的零样本测试。

本轮只训练两个多跳模型（残差、直接，各包含速度和每车道流量回归器），比较四种输出：

| 方法 | 固定规则 |
|---|---|
| B2_multihop | 残差预测 |
| Direct_multihop | 直接预测 |
| Fusion_mean50 | 残差、直接各 50% |
| Fusion_gap60mean | 可见速度观测间隔 >60 分钟或缺一侧用直接；其余各 50% |

间隔是目标所在位置前后可见速度观测之间的时间跨度。恰好 60 分钟用等权融合；缺任一侧用直接预测。速度和流量共用这个条件；不根据真值选择输出。流量融合使用保存预测中的整条 link veh/h，评分统一按车道归一化。规则在运行新走廊之前固定，不扫阈值、不调权重；二月保持留出。

## 缓存

沿用 `cache\task1` 的训练/评测特征和精确直接标签缓存，未修改特征缓存身份文件。缓存区分方向，不能复用 I5 的特征来替代 I405；当前 I405 尚无缓存，首次生成是正常的。

额外缓存两个模型及其原始预测。缓存身份包含样本、标签、参数、轮数、代码与依赖版本等；只有一致且内容哈希校验通过才 HIT。相同配置再次运行到新的结果目录，会复用特征、标签、模型、预测，只重算融合和评分。改变线程数也可能影响模型/预测缓存命中。

模型缓存保存 `residual`/`direct` 下的 `target_model_config.json`；继续隔离旧的仅残差推理入口。模型与原始预测实际位置见 `artifact_locations.json`，不重复复制到每个结果目录。两个融合预测保存在新结果目录。若中途停止，再次运行会复用已完整发布的缓存；未完成的临时缓存不会当作 HIT。

## 结果与日志

输出为 `runs\corridor_dev_时间戳`：

- **comparison_corridor.csv**：四种方法总分和两项 RMSE；分别相对残差、直接和等权融合的得分变化。
- `comparison_by_panel.csv`：按方向对比。
- `metrics_by_panel_regime.csv`：R1/R2/R3 分项；`metrics_summary.csv` 为正式聚合方式的总分。
- `evaluation\D7_I405_N\*_daily_errors.csv`、`*_gap_errors.csv`、`*_congestion_errors.csv`：四种方法的分日、缺口、拥堵误差。
- `evaluation\D7_I405_N\Fusion_*_predictions.parquet`：两个融合的预测。
- `cache_usage.json`：训练特征、评测特征、直接标签、模型、预测五类缓存的命中情况。
- `artifact_locations.json`：模型与两个原模型预测所在的缓存目录。
- `pairing_check.json`、`integrity_check.json`：样本/特征/参数配对、缓存内容及预测覆盖检查。
- `run_config.json`、`environment.json`、`input_manifest.json`：配置、代码、依赖和评测真值输入哈希；缓存 manifest 继续记录原始特征输入。
- `COMPLETE.json`、`log_update.json`：完成标记与日志更新状态。

得分越大越好，RMSE 越小越好。总分与描述性 RMSE 按 regime → 方向 → 走廊家族等权聚合。正式成功后自动把四种方法结果、相对增益、训练协议与缓存命中追加到项目根目录 **实验日志.txt**，保留历史并按结果目录去重。`log_update.json` 应为 `appended`。

日志追加失败可补记，无需重新训练：

```powershell
.\.venv\Scripts\python.exe -m task1 log-corridor --run '.\runs\corridor_dev_实际时间戳'
```

## 可选参数与检查

`-Panels` 可指定其他非 D12_I5 家族方向；本轮建议先保持默认。`-CacheDir` 指定项目内缓存位置，`-Output` 指定新的空结果目录，`-NoLog` 跳过日志。

小规模检查（14 天训练、3 天评测、300 样本/天、50 轮；不写正式日志）：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_corridor.ps1 -Mode smoke -Threads 4
.\.venv\Scripts\python.exe -m pytest tests -q
```

正式运行首次耗时主要来自新走廊的特征生成。控制台会显示日期进度和五类缓存 HIT/MISS；不能从小规模检查得出模型排名结论。
