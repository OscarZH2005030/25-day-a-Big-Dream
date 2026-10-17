# Task 1：残差预测 vs 直接预测

## 正式运行

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'
powershell -ExecutionPolicy Bypass -File .\run_task1_targets.ps1 -Mode dev -Threads 4 -ReferenceRun '.\runs\ablation_dev_20261016_174020'
```

一次完成六个模型（每个模型各训练速度和每车道流量回归器）：

| 特征组 | 残差版本 | 直接版本 |
|---|---|---|
| baseline，50 个特征 | B2_lightgbm | Direct_baseline |
| multihop，86 个特征 | B2_multihop | Direct_multihop |
| congestion，75 个特征 | B2_congestion | Direct_congestion |

每对模型的样本和特征完全相同，插值初值作为输入也保留。
两种目标均使用相同 L2 损失、固定 350 轮、原参数和随机种子：

- 残差：学习 y - initial；最终 initial + 模型输出。
- 直接：学习 y；最终使用模型输出，不加 initial。
- 速度标签为 km/h，流量标签为 veh/h/lane；提交输出乘回车道数。
- 两者使用相同的非负截断和评分逻辑。

训练仍为 2030/06–12，每天 1000 个样本；完整一月开发评测。
二月不在此命令中使用。结果只表示固定预算下的对比，尚未分别调参。

## 缓存如何复用

没有修改原特征构建及其缓存身份文件，沿用现有 `cache\task1`。
与上轮相同配置时，训练和评测特征应显示 `[cache] HIT`。
额外创建一个 `direct_labels` 缓存：按现有训练缓存中的样本键，读取原始
训练观测生成 float64 速度/每车道流量标签。不改变原抽样，也不通过
残差加回 float32 初值近似恢复标签。第一次显示 MISS，以后可 HIT。

三组残差对照会与 `-ReferenceRun` 的原实验逐 regime 校验；设置不一致、
样本或特征不一致、指标不能复现时均报错。缓存数据仍进行内容哈希和完整性验证。

## 查看结果与实验日志

输出位于新的 `runs\target_dev_时间戳`：

- **comparison_target_modes.csv**：三组残差/直接预测的配对分数、两项 RMSE、增益。
  `delta_direct_minus_residual > 0` 表示直接预测本次分数较高。
- `metrics_by_panel_regime.csv`：R1/R2/R3 分项。
- `evaluation\D12_I5_N\*_daily_errors.csv`、`*_congestion_errors.csv`、`*_gap_errors.csv`：细分误差。
- `pairing_check.json`：样本、参数、成对特征一致性检查。
- `reference_check.json`：三个残差对照的复现检查。
- `cache_usage.json`：训练、评测及直接标签缓存是否命中。
- `COMPLETE.json`：实验计算正常完成标记。
- **log_update.json**：日志状态，正式成功应为 `appended`，重复登记为 `already_logged`。

正式实验和复现校验完成后，自动追加到项目根目录的 **实验日志.txt**，
保留历史内容并按结果目录去重。若日志追加失败，模型和指标仍保留，
不用重跑训练，执行下面命令补记：

```powershell
.\.venv\Scripts\python.exe -m task1 log-target --run '.\runs\target_dev_实际时间戳'
```

小规模 smoke 仅用于验证，日志状态为 `skipped_smoke`，不混入正式实验结论。
需要明确不写日志时可加 `-NoLog`。

## 小规模检查与模型文件

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_targets.ps1 -Mode smoke -Threads 4 -ReferenceRun '.\runs\feature_ablation_smoke'
.\.venv\Scripts\python.exe -m pytest tests -q
```

模型位于 `models\特征组\residual或direct\方向`，通过
`target_model_config.json` 明确保存目标形式；比较流程按该形式预测。
这些模型使用专用元数据，避免旧的仅残差 predict 命令误将初值加到直接预测上。
本轮用于模型选择，正式 validation/private 推理接入将在选定方案后完成。
