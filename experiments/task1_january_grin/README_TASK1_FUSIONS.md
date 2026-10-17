# Task 1：多跳特征组的预测融合

正式运行（不重新训练模型）：

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'
powershell -ExecutionPolicy Bypass -File .\run_task1_fusions.ps1 -Mode dev -SourceRun '.\runs\target_dev_20261016_183242'
```

复用来源实验中的 `B2_multihop_predictions.parquet` 和 `Direct_multihop_predictions.parquet`，按完整目标键对齐，不依赖文件行顺序。沿用来源实验的方向和日期，读取评测特征缓存中的两侧可见观测间隔；不构建特征、不加载模型、不训练、不重新做模型推理。

一次比较五种方法：

| 方法 | 速度 | 流量 |
|---|---|---|
| B2_multihop | 原残差预测 | 原残差预测 |
| Direct_multihop | 原直接预测 | 原直接预测 |
| Fusion_channels | 残差预测 | 直接预测 |
| Fusion_mean50 | 残差、直接各 50% | 残差、直接各 50% |
| Fusion_gap30 | 间隔 ≤30 分钟用残差；其余用直接 | 使用同一个切换条件 |

间隔是同一位置前后两条可见**速度**观测的时间跨度。缺少任意一侧时用直接预测；恰好 30 分钟用残差。规则只依赖推理时可见信息；评测真值只用于评分和拥堵分组分析。30 分钟规则来自一月开发集观察，本轮属于开发比较，仍需其他走廊验证。此命令只使用来源实验的评测日期，默认来源不涉及二月留出集。

已有预测中的流量是整条 link 的 veh/h，融合在这个单位下进行；评分时沿用原来的按车道归一化。S_state 和描述性 RMSE 均按 regime → 方向 → 走廊等权聚合。RMSE 不是全体样本混合后的 RMSE。

运行前验证来源实验完成状态、配对状态及原复现检查，校验缓存文件和其原始 masked/network/template 输入哈希。读取原始评测真值重新评分，两个原模型必须逐 regime 复现原结果（容差 1e-7），否则报错且不记正式日志。不会因缓存缺失而重新生成缓存；应恢复来源实验使用的缓存，或用 `-CacheDir` 指向其现有位置。

结果保存到新的 `runs\fusion_dev_时间戳`：

- `comparison_fusions.csv`：五种方法的总分和两项 RMSE，以及分别相对残差、直接预测的变化。ΔS 大于 0 是提升，ΔRMSE 小于 0 是改善。
- `metrics_by_panel_regime.csv`：每个方向、R1/R2/R3 的结果。
- `metrics_summary.csv`：官方层级聚合的分数。
- `evaluation\方向\*_daily_errors.csv`、`*_gap_errors.csv`、`*_congestion_errors.csv`：五种方法的分日、缺口和拥堵误差。
- `evaluation\方向\Fusion_*_predictions.parquet`：仅保存三个新融合的预测；两个原模型文件继续复用来源目录。
- `prediction_reuse.json`：预测文件来源、复用缓存、目标数、切换到直接预测的数量、未重新训练标记。
- `reference_check.json`：两个原模型的分项复现检查。
- `run_config.json`、`environment.json`、`input_manifest.json`：规则、配置、代码和输入文件哈希。
- `COMPLETE.json`、`log_update.json`：计算完成与日志更新状态。

正式运行成功后，五种方法结果自动追加到根目录 `实验日志.txt`，保留历史并按结果目录去重。若日志写入失败，结果保留，执行下列命令补记，无需重算：

```powershell
.\.venv\Scripts\python.exe -m task1 log-fusion --run '.\runs\fusion_dev_实际时间戳'
```

可加 `-Panels D12_I5_N` 选择来源中已有的方向；`-NoLog` 关闭日志；`-Output` 指定项目内新的空结果目录。

小规模检查使用已有 `target_comparison_smoke` 的预测，不写正式实验日志：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_fusions.ps1 -Mode smoke
.\.venv\Scripts\python.exe -m pytest tests -q
```

若显式指定 `-Mode smoke -SourceRun ...`，会验证该来源的全部现有评测预测，但仍跳过正式日志。来源本身为抽样天数实验时，即使使用 dev 模式也不会记成正式实验。
