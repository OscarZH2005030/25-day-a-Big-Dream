# Task 2：独立数据管道、基准与图时空模型小检查

**新电脑 Codex 没有比赛聊天记录时，先读 [HANDOFF_CODEX.md](HANDOFF_CODEX.md)**，其中包含比赛背景、数据前置、完整执行顺序、回传清单及本次只做小检查的停止条件。

**当前只交付审计、代理数据准备和工程小检查，不启动正式训练。** 用户已确认原研发计划，正式运行电脑为 RTX 4070、i9-13900HX、32GB RAM。原电脑 Task 1 的代码、缓存、环境、权重和正式日志均不需要修改。

已有仓库时先在仓库根目录 `git pull --ff-only`。本目录为 `experiments/task2_queue_20261024`，与旧 Task 1 实验目录相互独立。

## 1. 当前已经实现什么

- 全部八方向、train/validation/private 的窗口、历史、模板、阈值、拓扑和质量审计；每 split 80 窗、87,000 输出行。
- 273 天冻结日期表：[FROZEN_SPLIT.json](FROZEN_SPLIT.json)。6—11 月 fit，12 月 tune，1 月 calibrate，2 月 audit；普通入口不会读取后两个月的代理未来标签。
- 从公开 train 观测构造同日滚动代理窗，13 个输入位置为 T−60…T，其中 T 只用 masked 公开观测；未来标签为 T+5…T+30 的六步，质量阈值为75%。
- 按公开文件内容 SHA256、日期和 schema 缓存日数组；原子发布、读取校验、损坏重建、相对路径索引。复跑新输出目录可复用相同 `cache/`。
- 精确 T−5／eligible 的官方 persistence、历史有效值携带的 T−5/T 增强版本、轻量传播外推、12轮 LightGBM 小检查。
- 共享时间 Transformer＋稀疏有向图注意力＋传播延迟先验＋onset/ongoing 条件头；一次输出六步，孤立节点保留自身时间路径。
- 窗口等权 BCE＋soft-IoU 损失、逐层宏平均评分、未来扰动检查、断点保存/重载，以及官方真实窗口的推理和模板检查。

主模型是新的原型，不是原版 PDFormer 的复现；当前标准配置 width=128、layers=4、heads=4，**1,407,243 个参数**。辅助速度头已保留接口，本轮不拟合；匝道分支、CTM、完整正式训练/选择/校准/提交入口在后续阶段实现，本版本不宣称它们已经完成。

## 2. 任务边界与结果含义

Task 2 只评 D7_I10、D7_I210、D7_I405、D12_I5 的双向，**不包含 D12_I405**。每窗真实 history 是十二帧 T−60…T−5；可选 T 帧来自主线 masked 发布表，缺失保持缺失。

**官方 queue_true 在所有 split 都不公开。** 代理标签来自含噪公开 train 速度，不能叫官方真值；所有分数都标为 proxy/engineering，不能拿来证明 Kaggle 排名。小检查只用少量样本、固定0.5阈值，不用于模型选择。

公开 `condition` 可作输入，`future_coverage`、未来质量、未来匝道和 Task 1 全天离线预测不能作输入。预测不会根据未知未来质量删行。标签缺失不是“无队列”。准确率和全局 cell-micro IoU 不能替代逐窗、condition、方向、家族等权的评分。

## 3. 下载代码和准备数据

```powershell
git clone https://github.com/OscarZH2005030/25-day-a-Big-Dream.git
cd .\25-day-a-Big-Dream\experiments\task2_queue_20261024
```

官方数据需要另行下载或从原电脑复制。Git 中没有原始数据、虚拟环境、日缓存、工程权重或训练标签。`ReleaseRoot` 应指向含 `config/`、`corridors/`、`task2/` 的 **kaggle_public 文件夹本身**，不是其父目录：

```powershell
$ReleaseRoot = 'D:\datasets\2026-ieee-big-data-traffic-flow-bench\kaggle_public'
```

请替换成真实路径，支持中文和空格。可以与 Task 1 共用只读的原始官方数据，但不能把 Task 1 特征缓存当作 Task 2 输入。

## 4. RTX 4070 建立独立环境

安装 Python 3.14（参考版本3.14.8），随后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_task2.ps1
```

可用 `-PythonPath 'C:\路径\Python314\python.exe'` 指定解释器。脚本只在本目录创建 `.venv-task2`，安装锁定依赖与 PyTorch `2.10.0+cu128`，核对源码指纹、冻结划分、GPU，并运行 CPU 单元检查。不会安装系统 CUDA Toolkit，也不会操作 `.venv-grin` 或 Task 1 目录。

4070 的实际显存由环境检查报告，不按名称猜测。默认按保守配置使用 batch=1、4个线程、无 DataLoader 子进程；32GB 内存下按日处理，不把273天所有方向的张量拼入内存。

如已建立完全匹配的独立环境，可检查：

```powershell
.\.venv-task2\Scripts\python.exe -B .\check_environment.py --device cuda
```

如果 CUDA 不可用，检查 NVIDIA 驱动与安装日志，不要静默改成正式 CPU 训练或升级整套依赖。CPU 调试可在下面命令显式使用 `-Device cpu`。

## 5. 另一台电脑按这个顺序运行

以下都是工程检查，不是正式训练。每个 `Output` 必须是新的空目录；已存在则换成 `002` 等新名字。相同缓存自动复用。

```powershell
# A. 全部八方向、三split审计；只读小表，不读取2月未来代理标签。
powershell -ExecutionPolicy Bypass -File .\run_task2.ps1 -Mode audit -ReleaseRoot $ReleaseRoot -Device cpu -Output '.\runs\audit_4070_001'
if ($LASTEXITCODE -ne 0) { throw 'Audit failed: inspect audit.json before proceeding.' }

# B. 八方向小规模代理数据：每role最多2天，每方向每role最多8窗。
#    只准备fit/tune；最多128个代理窗，不拟合任何模型。
powershell -ExecutionPolicy Bypass -File .\run_task2.ps1 -Mode prepare -ReleaseRoot $ReleaseRoot -Device cpu -Output '.\runs\prepare_4070_001'
if ($LASTEXITCODE -ne 0) { throw 'Proxy preparation failed: inspect data/schema before smoke.' }

# C. 4070工程小检查：默认I210东＋D7_I405南，神经网络最多4次更新，树12轮。
#    另检查最大352节点图的前向，以及两类官方窗在三split上的输入/输出。
powershell -ExecutionPolicy Bypass -File .\run_task2.ps1 -Mode smoke -ReleaseRoot $ReleaseRoot -Device cuda -Output '.\runs\smoke_4070_001'
if ($LASTEXITCODE -ne 0) { throw 'GPU smoke failed: do not start further training.' }
```

`MaxDays`、`MaxWindows`、`Stride`、`Steps`、`Threads` 可显式指定；smoke 硬限制为最多2天/role、8窗/方向/role、4次神经更新。图网络与 LightGBM 的这几次拟合仅验证工程链路，不代表正式训练开始。

运行完成后先停在这里，把以下报告交给原电脑的 Codex：

```text
runs/audit_4070_001/audit.json
runs/prepare_4070_001/dataset/index.json
runs/smoke_4070_001/SMOKE_COMPLETE.json
runs/smoke_4070_001/runtime.json
runs/smoke_4070_001/engineering_proxy_diagnostics.json
runs/smoke_4070_001/official_window_schema_checks.json
```

合格标记包括 `passed=true`、`formal_training_started=false`、`calibrate_or_audit_labels_opened=false`、缓存复用和未来扰动检查通过。这里的 `official_window_schema_checks` **只验证格式和输入，不含隐藏真分**。

本版本没有 `train`／`dev`／完整提交命令，传入这些 action 会拒绝。下一步应分析审计和小检查报告，再编写正式实验入口，不自动进入正式训练。

## 6. 数据准备、缓存与恢复

默认缓存为本目录 `cache/`；输出在本目录 `runs/`，均被 Git 忽略。`-CacheDir` 和 `-Output` 只允许本 Task 2 工程内部的独立路径，避免误写原数据、Task 1 或代码目录。

准备与小检查中断可按 Ctrl+C；重跑用新输出目录，复用同一缓存。日缓存写入未完成只会导致 cache miss，完整缓存有 SHA256 校验；工程小检查的四次更新从头做，不承诺正式训练断点续跑。不要手改冻结划分、cache sidecar 或完成标记绕过检查。

索引使用相对缓存路径，复制整个 Task 2 输出＋cache 到相同相对结构后可加载。源码变化、原始数据变化或不同schema需重新生成索引；不要跨 schema 强行复用旧缓存。它不会读取 Task 1 的现有缓存。

高级数据准备接口还有显式 `-Mode prepare -Scope full-inputs`，只构建 fit/tune 的全部候选索引与日缓存，不训练，也不打开1/2月未来标签；默认 stride=6（30分钟原点间隔）。这会读取较多数据并占用额外磁盘，**本次接手只运行上面的有限小检查命令**。正式训练的均衡采样、稀疏评估、三种子选择与校准仍按计划后续实现。

## 7. 本次验证与局限

本机参考环境：Python 3.14.8、PyTorch 2.10.0+cu128、RTX 3060 Laptop 6GB。RTX 4070 尚需在目标电脑执行同样的小检查，不能把3060实测写成4070实测。

验收摘要与源码指纹分别见 [verification/acceptance.json](verification/acceptance.json) 和 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。CPU 检查覆盖代理标签边界、质量阈值、因果切片、保留月份拦截、缓存迁移/损坏、图方向/孤立节点、损失梯度、精确官方 persistence、宏平均和模板完整性。

GPU 小检查检查标准128×4原型的有限梯度、四次更新、最大352节点图前向、模型保存重载和12个官方窗口格式。它不计算官方 S_queue，不打开1/2月代理监督，不拟合融合/阈值，不说明模型已经优于基准。

## 8. 官方依据与后续交接

- [任务与在线输入边界](https://github.com/jacky850/trafficflowbench-public)
- [评分与标签不可用说明](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md)
- [官方T−5 persistence与T行可用性](https://github.com/jacky850/trafficflowbench-public/blob/main/src/task2/build_task2_persistence_submission.py)
- [原研发计划](PLAN_TASK2_QUEUE.txt)

报名是否完成仍需用户核实：官方要求10月25日23:59 AoE前邮件报名；本代码不会发送邮件或提交 Kaggle。

给另一台电脑 Codex 的指令：

> 请拉取main，进入experiments/task2_queue_20261024并完整阅读README。保持Task1不变，使用独立.venv-task2。先核对4070实际显存、Python与数据根目录，依次执行setup、audit、有限prepare和smoke。只做工程检查，不运行full-inputs、不编写/启动正式训练、不打开1/2月未来代理标签。把上述六类报告交回，继续明确区分proxy指标与官方真分。
