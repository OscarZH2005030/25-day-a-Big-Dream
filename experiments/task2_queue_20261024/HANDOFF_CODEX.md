# 给没有比赛历史记录的另一台电脑 Codex

交接日期：2026-10-24。本文足以从零接手，不需要读取原聊天或原电脑磁盘。用户已经授权本阶段的环境安装、数据审计、有限代理样本准备和 RTX 4070 工程小检查；**没有授权正式训练**。

## 1. 比赛背景和本次目标

我们参加 2026 IEEE Big Data–Traffic Flow Bench，目标是做有竞争力、可复现的方案。比赛入口：<https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench>；官方工具：<https://github.com/jacky850/trafficflowbench-public>。

比赛包含 Task1 状态补全、Task2 队列预测、Task3 物理一致性和 Task4 OD/路径流量估计。当前只接手 **Task2 的工程准备**。仓库里其他 Task1 版本及已完成结果是历史资产，不是本次启动入口；不要运行它们的训练脚本，不要修改其代码、进程、环境、缓存、模型、日志或结果。原电脑 Task1 续跑队列已于本次交接前显示 completed；本机状态仍需独立核对，不能据此终止任何现有程序。

Task2 研究路线为传播延迟感知的图时空 Transformer，加 LightGBM 互补专家，后续才考虑物理专家、正式训练与选择。本次代码已经实现前置工程与小检查，不意味着已经训练好最强模型。Task1 的历史 0.936910 等成绩不是 Task2 的目标线，也不是比赛总分。

## 2. 从哪里开始读

仓库：`https://github.com/OscarZH2005030/25-day-a-Big-Dream.git`，分支：`main`。

目录：`experiments/task2_queue_20261024`。工程代码首发提交为 `e007126677d17becc3d44840020674da1ece9df0`；本次交接文档在其后提交，**拉取 main 最新版，不 checkout 回旧提交**。

依次阅读：

1. 本文：背景、授权范围、执行和停止条件。
2. 同目录 [README.md](README.md)：环境、数据路径、命令、输出含义和限制。
3. [PLAN_TASK2_QUEUE.txt](PLAN_TASK2_QUEUE.txt)：完整研发路线；其中后续正式阶段不属于本次执行范围。
4. [FROZEN_SPLIT.json](FROZEN_SPLIT.json)、[verification/acceptance.json](verification/acceptance.json)：冻结协议与原电脑验收证据。

## 3. 必须理解的数据与评分边界

- 八个有效方向：D7_I10_E/W、D7_I210_E/W、D7_I405_N/S、D12_I5_N/S。D12_I405_N/S 不参与 Task2。
- 每方向每 split 有 onset/ongoing 各5窗；每 split 合计80窗、87,000个未来输出cell。validation＋private最终需174,000行，但本阶段不生成完整正式提交。
- 公开 history 是12帧 T−60…T−5；T帧仅使用该时刻确已公开的 masked 主线观测，可缺失。预测 T+5…T+30 共六步。所有预测特征必须≤T。
- 官方 queue_true 在所有 split、包括 train 都没有公开。公开 train 观测构造的是含噪 proxy 标签；模板全零是占位，不能当标签。不要尝试寻找或伪造隐藏真值。
- 未来标签缺失/低质量保持监督 mask，不能标为“无队列”。公开 condition 可使用；future_coverage、未来质量、未来匝道、Task1 全天离线预测均不能作输入。
- 评分为逐窗口时空 IoU，再按 condition、方向、家族等权聚合；accuracy 或全局 cell-micro IoU不能代替。本地 proxy 不能称为官方 S_queue 或奖项排名。
- 冻结训练日历是合成数据日期：2030年6—11月 fit、12月 tune、2031年1月 calibrate、2月 audit。它不是本机当前年月。
- 本次只读取 fit/tune 的代理监督。`-Mode audit` 是文件结构审计，不代表获准打开2月 `audit` 角色的未来代理标签。

## 4. 目标电脑与前置条件

目标是 Windows＋RTX 4070，Intel i9-13900HX，32GB RAM。先核对实际系统、驱动、GPU和显存；4070的显存不凭型号猜测。原电脑是3060，4070尚未实测。

使用独立 Python3.14 环境（参考3.14.8）、本目录 `.venv-task2` 和锁定 PyTorch2.10.0+cu128。不要复用或修改 Task1 的 `.venv-grin`、Strong v4 的Python3.12环境，或全局升级依赖。安装脚本无需安装系统CUDA Toolkit。

数据没有上传GitHub，需要从原电脑复制公开数据，或由用户通过官方Kaggle入口取得并解压。`ReleaseRoot` 指向含以下资产的 **kaggle_public 本身**：

```text
config/competition_contract.json
config/synthetic_release_v1.json
corridors/<有效方向>/network/{links.csv,fd_parameters.csv,lwr_mainline_topology.csv,...}
corridors/<有效方向>/train/mainline_states/...
corridors/<有效方向>/{train,validation,private}/mainline_states_masked/...
task2/<有效方向>/{train,validation,private}/window_index.csv
task2/<有效方向>/{train,validation,private}/window_history.parquet
task2/<有效方向>/{train,validation,private}/sample_submission_queue.csv
```

先定位本机已有数据。不要照抄原电脑 C 盘路径，也不要把示例 D 盘路径当真实路径。找不到数据、Python、权限或下载所需凭据时，先完成其余独立核对，再报告缺项并向用户索取；不要把不完整数据当完整审计通过。只需要公开原始数据，不需要旧 Task1 缓存、权重或原聊天。

## 5. 实际执行顺序

如果尚未克隆，在用户选定的工作目录执行：

```powershell
git clone https://github.com/OscarZH2005030/25-day-a-Big-Dream.git
cd .\25-day-a-Big-Dream
```

已有仓库则定位仓库根目录，先检查 `git status`，保留用户本地修改，再 `git pull --ff-only`。若拉取有冲突，诊断并保留工作，不用 reset/clean/force 覆盖。

记录当前提交，再进入实验目录：

```powershell
git log -1 --format="%H %s"
cd .\experiments\task2_queue_20261024
$ReleaseRoot = 'D:\datasets\2026-ieee-big-data-traffic-flow-bench\kaggle_public' # 替换成实际路径
```

下列每步成功后再做下一步：

```powershell
# 独立环境、源码指纹/GPU检查与CPU单元检查。
powershell -ExecutionPolicy Bypass -File .\setup_task2.ps1
if ($LASTEXITCODE -ne 0) { throw 'Setup failed; inspect before proceeding.' }

# 单独留存GPU环境检查结果，含实际显存。
.\.venv-task2\Scripts\python.exe -B .\check_environment.py --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }

# 八方向、三split结构审计；不打开1/2月未来代理监督。
powershell -ExecutionPolicy Bypass -File .\run_task2.ps1 -Mode audit -ReleaseRoot $ReleaseRoot -Device cpu -Output '.\runs\audit_4070_001'
if ($LASTEXITCODE -ne 0) { throw 'Audit failed.' }

# 有限代理样本，不是full-inputs；默认最多128窗，仅fit/tune。
powershell -ExecutionPolicy Bypass -File .\run_task2.ps1 -Mode prepare -ReleaseRoot $ReleaseRoot -Device cpu -Output '.\runs\prepare_4070_001'
if ($LASTEXITCODE -ne 0) { throw 'Proxy preparation failed.' }

# 两代表方向的4070工程小检查；最多4次网络更新、12轮小树。
powershell -ExecutionPolicy Bypass -File .\run_task2.ps1 -Mode smoke -ReleaseRoot $ReleaseRoot -Device cuda -Output '.\runs\smoke_4070_001'
if ($LASTEXITCODE -ne 0) { throw 'GPU smoke failed.' }
```

若 Python3.14 未被 py 启动器找到，按 README 给 setup_task2.ps1 传 `-PythonPath`。默认 batch=1、width=128、4层、4线程，无DataLoader子进程。优先保持默认预算，不为了4070算力增大更新次数。

输出目录必须是新的空目录。`001` 已用过则统一换新后缀，不覆盖旧报告。沿用同一Task2 `cache/` 即可复用缓存；无需清理Task1。中断后用新输出重跑有限检查，四步工程拟合从头做，不把工程 checkpoint 当正式训练断点。

## 6. 验收、回传和停止条件

原电脑参考验收：38项CPU检查通过；八方向结构审计1450 links、315孤立links、240窗、261000模板行（这是三split合计）；有限代理准备128窗；3060完成标准模型四次更新、最大352节点图前向和12个真实官方窗格式检查。

目标电脑不要求复现完全相同耗时或GPU逐位浮点值，需满足实际检查与配置一致。正确输出应有：

- `passed=true`、`formal_training_started=false`。
- `calibrate_or_audit_labels_opened=false`。
- 缓存复用、未来扰动不改变输入/预测、checkpoint重载检查通过；loss与梯度有限。
- `official_queue_score=null`，任何本地分数仍明确是工程/代理诊断。

回报给用户：当前commit及工作区是否有修改、Python/torch/驱动/GPU实际显存、逐步通过或失败状态、耗时/峰值显存、任何代码或环境修复，以及以下文件的真实绝对路径：

```text
runs/audit_4070_001/audit.json
runs/prepare_4070_001/dataset/index.json
runs/smoke_4070_001/SMOKE_COMPLETE.json
runs/smoke_4070_001/runtime.json
runs/smoke_4070_001/engineering_proxy_diagnostics.json
runs/smoke_4070_001/official_window_schema_checks.json
```

报告文件保留在本机，不擅自上传原始数据、缓存、凭据或大模型到Git。若缺数据或检查失败，保留错误输出，诊断后报告阻塞项；不要为“通过”而修改冻结划分、预算、阈值、哈希或跳过测试。

**完成上述检查就停止并向用户汇报。** 不运行full-inputs、不新增或启动正式训练、不打开1/2月未来代理监督、不调榜、不提交Kaggle、不发送邮件。当前版本也没有正式训练action，工程四步checkpoint不是获奖候选模型。

## 7. 获奖手续提示

截至2026-10-24，用户尚未确认邮件报名完成。官方要求10月25日23:59 AoE前登记；如仍在有效时间内，提醒用户核实官方页面。这里只提醒，不代表已报名，也不授权代发邮件。具体手续及最终提交要求以比赛页面为准。
