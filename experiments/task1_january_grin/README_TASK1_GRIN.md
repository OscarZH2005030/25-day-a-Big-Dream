# Task 1：GRIN 第一轮

## 安装与运行

在项目根目录打开 PowerShell：

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'
powershell -ExecutionPolicy Bypass -File .\setup_grin.ps1 -TorchBuild cu128
powershell -ExecutionPolicy Bypass -File .\run_task1_grin.ps1 -Mode smoke -Device cuda -BatchSize 2
```

安装脚本创建独立 `.venv-grin`，沿用已固定的基础依赖并安装 PyTorch 2.10.0 CUDA 12.8 版。官方 wheel 索引提供当前 Python 3.14 的 Windows 包。脚本检查 GPU 可用性并运行一次 CUDA 矩阵计算。本机已检测到 RTX 3060 6GB；默认隐藏维度 32、窗口 48、batch 4。GPU 版下载较大。使用预编译 wheel，不要求单独安装 CUDA Toolkit，也不需要安装 PyG、torch-scatter、Lightning 或下载官方数据集。

小规模检查完成后正式运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_grin.ps1 -Mode dev -Device cuda -BatchSize 4
```

若显存不足，将 `-BatchSize 4` 改为 `-BatchSize 2` 或 `1`，使用新的结果目录。需要 CPU 检查时安装 `setup_grin.ps1 -TorchBuild cpu`，运行时明确选择 `-Device cpu`；再次执行 CUDA 安装命令即可切换版本。原 `.venv` 与 baseline 依赖不变。

## 模型与数据协议

本实现依据 [GRIN 官方仓库](https://github.com/Graph-Machine-Learning-Group/grin) 和 [论文](https://arxiv.org/abs/2108.00298)，独立实现核心架构，并适配本比赛：

- 前向、反向 GRIL 各包含：上一时刻隐状态的第一阶段预测、自环排除的空间解码、第二阶段预测、图扩散 GRU 状态更新。
- 两个方向的表示通过 MLP 融合；一层循环、扩散阶数 2、空间解码阶数 1、可学习初始隐状态；无全局注意力、无额外输出节点嵌入。
- 输入速度和每车道流量及各自可见掩码，附加日内/星期正余弦时间特征。训练输出为直接预测，不与 LightGBM 或插值混合。
- 以观测站为节点，按主线拓扑连接最近的上下游观测站，跨过无传感器主线路段；按物理距离倒数赋权，保留正向和反向归一化支持，不使用匝道或连接器作为观测节点。

这是 GRIN 在 TrafficFlowBench 上的首个适配版本，并非原论文实验逐项复现。损失、时间协变量、窗口和训练预算都明确记录在配置里。

固定协议：

| 内容 | 配置 |
|---|---|
| 方向 | D12_I5_N，258 个观测站 |
| 训练 | 2030-06-01 至 2030-12-31，214 天 |
| 开发评测 | 完整 2031 年 1 月，550,149 个目标 |
| 留出 | 二月保持不用 |
| 训练监督样本 | 每天 1000 个原 Task 1 缺失目标，seed=20261016，沿用 baseline 的按天 RNG 抽样顺序 |
| 模型输入 | masked 文件；不向模型传入隐藏真值、真实拥堵标签或 imputed 值 |
| 标准化 | 仅拟合训练期 `pct_observed >=75` 的可见观测；评测复用训练统计量 |
| 损失 | 评分尺度加权 MSE，速度 /25、每车道流量 /600，权重 0.54/0.46；方向第一/第二阶段辅助损失权重 0.1 |
| 额外随机遮挡 | 每步对 5% 的可见节点随机遮挡，用原可见值监督；原有缺失始终保留 |
| 默认预算 | 最多 20 epochs，每 epoch 最多 100 次优化步骤，Adam lr=0.001，patience=5 |
| 时间窗口 | 48 个 5 分钟时刻，共 4 小时；stride=24，重叠区域用正三角权重合并；不跨午夜 |
| checkpoint | 根据一月开发 S_state 选最佳；一月真值不参与梯度更新或标准化 |

训练会抽取图时间窗口，同一训练目标可能出现在多个窗口；训练还包含随机遮挡的可见观测。数据划分、原缺失监督样本和评分与 baseline 对齐，训练预算和上下文不同，不宣称等算力对比。窗口长度限制了 GRIN 可见上下文，是第一轮配置选择。

若原正式训练缓存存在，会额外检查抽样 ID 与原 baseline 完全相同，结果为 `sample_check.json`。正式评测检查每个 regime 的目标数与冻结 baseline 一致。

## 缓存与结果

GRIN 使用独立 `cache\grin`，保存每日图张量、节点顺序、图结构、标准化统计和训练抽样键；现有 LightGBM 表格特征缓存与此输入格式不同，不直接用于 GRIN。重复运行相同数据配置会 HIT，无需重新读原始数据构造张量。

评测缓存仅含 masked 输入与掩码，不保存一月真值。正式与小规模缓存按日期和抽样分开。缓存对原始文件、代码、依赖版本做哈希校验。

运行结果：`runs\grin_dev_时间戳`。

- **comparison_vs_baseline.csv**：GRIN 与四种原多跳方法比较，重点看 `delta_S_state_vs_frozen_baseline`；冻结方法是 `Fusion_gap60mean`，原得分 0.928029。
- `metrics_by_panel_regime.csv`、`metrics_summary.csv`：直接调用 `task1.metrics.score/aggregate`；速度与每车道流量评分及 R1/R2/R3 等权方式不变。
- `GRIN_predictions.parquet`：完整目标预测，输出流量已乘回车道数；附带缺口分组以便分析。
- `GRIN_daily_errors.csv`、`GRIN_congestion_errors.csv`、`GRIN_gap_errors.csv`：分日、拥堵、缺口误差。
- `best.pt`、`training_history.csv`、`selection.json`：最佳权重、各 epoch 开发得分与选择信息。
- `run_config.json`、`cache_usage.json`、`environment.json`：训练参数、缓存、PyTorch/设备/代码指纹。
- `COMPLETE.json`、`log_update.json`：完成及日志状态。

正式完成后自动追加到根目录 `实验日志.txt`，包含 GRIN 得分、RMSE 和相对冻结 baseline 的提升，按结果目录去重。smoke 不写正式实验结论。日志写入失败时补记：

```powershell
.\.venv-grin\Scripts\python.exe -m task1.grin log --run '.\runs\grin_dev_实际时间戳'
```

小规模模式仅训练前 7 天、评测前 3 天、300 个监督样本/天、2 epochs、每 epoch 3 步、hidden=16、window=24。用于验证流程，不能用其分数判断 GRIN 是否超过正式 baseline。

中途停止但已有 `best.pt` 时，可以只评测该 checkpoint，无需重训：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1_grin.ps1 -Device cuda -Checkpoint '.\runs\grin_dev_实际时间戳\best.pt'
```

这会读取 checkpoint 的训练配置与缓存身份，写入新的结果目录。不是续训。仅准备正式张量可加 `-PrepareOnly`。调整训练预算可用 `-Epochs`、`-StepsPerEpoch`；首轮建议保持默认。查看测试：

```powershell
.\.venv-grin\Scripts\python.exe -m pytest tests -q
```

安装依据：[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)、[官方 CUDA 12.8 wheel 索引](https://download.pytorch.org/whl/cu128/torch/)。
