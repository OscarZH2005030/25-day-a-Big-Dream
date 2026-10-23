# 给另一台电脑 Codex 的实验交接说明

用户已确认：**取消精简方案，继续原完整对照方案。** 不跳过 T0、原 T1 或原固定融合，不改变划分、评分、训练预算、融合权重和孤立节点回退。本文是迁移交接说明；安装与原完整协议见同目录 [README.md](README.md)。

## 先读：这次还要做什么

目标是补齐十个方向的 T1／动态缺失实验，再比较单模型和固定融合，判断相对历史本地分数 `0.936910312032` 是否提升。

**本次接手默认只补未完成方向，不从六方向默认队列重新开始。** 代码、旧四方向参考表已在 GitHub；官方数据、当前电脑的模型、缓存、断点、I210 本次结果没有上传。本文没有附带这些产物。

仓库：`https://github.com/OscarZH2005030/25-day-a-Big-Dream.git`，分支：`main`。

工作目录：`experiments/task1_t1_dynamic_20261021`。训练源码版本为提交 `83237888343d650f7e5a0eb4597e9a6fa52b5d92`，本次交接只更新文档。

## 进度快照：2026-10-22 21:31 PDT / 2026-10-23 04:31 UTC

**这是有时间戳的快照，原电脑当时仍在训练，不能把本文当实时状态。**

| 方向 | 快照状态 | 接手任务 |
|---|---|---|
| D7_I405_N、D7_I405_S | 已完整完成 | 复用，不重训；Git 中有固定参考表 |
| D12_I5_N、D12_I5_S | 已完整完成 | 复用，不重训；Git 中有固定参考表 |
| D7_I210_E、D7_I210_W | 已完整完成，校验通过，实验日志已追加 | 复用，不重训；结果仍在原电脑，Git 中未包含 |
| D7_I10_E | T1 正在训练，第 13 轮第 1 步；第 12 轮已完成 | 核对最新状态；此方向 T1、A_dynamic、T0 尚未全部完成 |
| D7_I10_W | 尚未开始 | 完整训练 T1、A_dynamic、T0，保留共享 LightGBM 和全部对比 |
| D12_I405_N、D12_I405_S | 尚未开始 | 完整训练 T1、A_dynamic、T0，保留共享 LightGBM 和全部对比 |

因此快照时已完成 **6/10 方向**。本轮新增六方向的 **18 个神经模型已完成 6 个**，剩余 12 个中有 1 个正在训练。LightGBM、预测和评分不计入这 18 个神经模型。

I210 的完成标记是真实的 `COMPLETE.json`，不是仅根据日志推断；批次完成耗时约 35,723 秒，这是该次续跑调用的耗时，不能当成六个模型从零训练的总耗时。

## 原电脑产物位置与接手前检查

以下为原电脑绝对路径，在新电脑上不一定存在：

```text
项目：C:\Users\DELL\Desktop\UW\IEEE比赛
队列：C:\Users\DELL\Desktop\UW\IEEE比赛\runs\remaining_dynamic_dev_20261021_141813_814491
I210：上述队列\batches\01_D7_I210
I10： 上述队列\batches\02_D7_I10
I405：上述队列\batches\03_D12_I405（快照时尚不存在）
缓存：C:\Users\DELL\Desktop\UW\IEEE比赛\cache\transformer_cross_corridor
日志：C:\Users\DELL\Desktop\UW\IEEE比赛\实验日志.txt
运行输出：C:\Users\DELL\Desktop\UW\IEEE比赛\runs\_queue_control\resume_20261022_111918\stdout.log
错误输出：同一目录下的 stderr.log
```

已完成旧四方向的原始目录：

```text
...\runs\transformer_training_dev_20261019_183939_881805
...\runs\d12_i5_dynamic_dev_20261020_124730_964790
```

接手 Codex 应先做以下检查：

1. 读取本文和同目录 README，检查新电脑系统、GPU、Python 与锁定依赖。
2. 获取原电脑最新的队列 `status.json`、各批次 `status.json` 和 `COMPLETE.json`，确认哪些方向已经完成。无法访问原电脑时，向用户确认最新进度和两台机器的分工，不猜测实时状态。
3. 核对现有缓存、断点和官方数据是否实际传到了新电脑。只克隆仓库并不等于断点迁移。
4. 若新电脑接手 I10＋I405，安排原电脑在安全断点停止该队列，再复制稳定产物。若原电脑继续 I10，新电脑只跑 `-Batch i405`；还须安排原电脑的队列在进入 I405 前停止，避免两台重复训练。不要直接强杀、不相关地取消其他任务，或删除原产物。
5. 以下默认命令适用于新电脑从空缓存补跑“仍未完成的批次”。如果原电脑 I10 已完成，则跳过全部 I10 命令，只做 I405。

## 必须保持的完整实验协议

- 每个待跑方向训练 **T1、A_dynamic、T0** 三个神经模型，另保留共享 LightGBM。不要执行此前讨论过的精简方案。
- 每方向比较 `T0`、`T1`、`A_dynamic`、`LightGBM`、`Current_fusion`、`A_dynamic_fixed`。
- `A_dynamic` 指 T1＋动态缺失；所谓“动态融合”是动态缺失模型参加的固定权重融合，不是动态学习融合权重。
- 连通节点两通道原融合：`45% T0 + 45% T1 + 10% LightGBM`。
- 连通节点两通道新融合：`45% T0 + 45% A_dynamic + 10% LightGBM`。
- **孤立节点两通道均精确回退到共享 LightGBM。**
- 保持 28 天历史、168 天训练、22 天内部验证、55 天留出及原日期表。
- 正式预算最多 **60 轮 × 1024 步**，patience=8；保留原 batch=2、种子、网络、采样、优化器和动态缺失配置。更强显卡也不加预算。
- 评分为速度 RMSE /25、每车道流量 RMSE /600，权重 0.54／0.46，按 regime→方向→家族等权聚合。
- 模型和内部产物冻结后才进入对应批次留出评测，不用留出结果修改本轮预算、权重或模型选择。
- 记录新电脑 GPU、驱动和依赖。不同 GPU 的浮点结果可能不同，不宣称跨显卡逐位一致。

## Windows＋NVIDIA：获取代码、建立环境

在 PowerShell 中运行；已有仓库先确认没有需保留的本地修改，再 `git pull --ff-only`。

```powershell
git clone https://github.com/OscarZH2005030/25-day-a-Big-Dream.git
cd .\25-day-a-Big-Dream\experiments\task1_t1_dynamic_20261021
powershell -ExecutionPolicy Bypass -File .\setup_team.ps1
```

需要 Python 3.14（原电脑 3.14.8），脚本使用锁定依赖和 PyTorch `2.10.0+cu128`。不要因为新电脑环境安装困难就静默升级或更换训练库；先报告不匹配项。

下载或复制官方 `kaggle_public` 数据。可以放在 README 的默认目录，或使用外部目录。以下示例在当前 PowerShell 会话定义实际数据路径，先替换成新电脑真实位置：

```powershell
$ReleaseRoot = 'D:\datasets\2026-ieee-big-data-traffic-flow-bench\kaggle_public'
.\.venv-grin\Scripts\python.exe -B .\check_environment.py --device cuda --release-root $ReleaseRoot
```

数据目录中的 `task1/` 与代码目录中的 `task1/` 是两个不同目录，不要互相覆盖。数据、模型、缓存、运行目录和个人凭据不提交 Git。

## 先小检查，再只跑剩余批次

以下使用两个独立顶层输出目录，一批结束后再启动下一批，避免在一张 GPU 上争抢资源。**不要省略 `-Batch`**：默认 `all` 会从已完成的 I210 开始。

接手两批时，先分别小检查；如果仅接手 I405，只运行 I405 的检查和正式命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode smoke -Batch i10 -Device cuda -ReleaseRoot $ReleaseRoot -Output '.\runs\handoff_i10_smoke_001'
if ($LASTEXITCODE -ne 0) { throw 'I10 smoke failed; inspect before formal training.' }

powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode smoke -Batch i405 -Device cuda -ReleaseRoot $ReleaseRoot -Output '.\runs\handoff_i405_smoke_001'
if ($LASTEXITCODE -ne 0) { throw 'I405 smoke failed; inspect before formal training.' }
```

小检查通过后正式运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -Batch i10 -Device cuda -ReleaseRoot $ReleaseRoot -Output '.\runs\handoff_i10_formal_001'
if ($LASTEXITCODE -ne 0) { throw 'I10 failed; resume this batch before proceeding.' }

powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -Batch i405 -Device cuda -ReleaseRoot $ReleaseRoot -Output '.\runs\handoff_i405_formal_001'
if ($LASTEXITCODE -ne 0) { throw 'I405 failed; inspect and resume.' }
```

单批正式队列正常结束会有顶层 `BATCHES_COMPLETE.json`，其中 `ten_direction_complete=false` 是预期状态；其 `batches/02_D7_I10` 或 `batches/03_D12_I405` 中必须另有通过验收的 `COMPLETE.json`。这不是完整十方向完成标记。

如果以上输出目录已经存在且有配置，先查状态；续跑使用同一批次、同一设备、同一数据/缓存路径和同一输出目录，加 `-Resume`。例如：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -Batch i10 -Device cuda -ReleaseRoot $ReleaseRoot -Resume -Output '.\runs\handoff_i10_formal_001'
```

中断时按 Ctrl+C 并等待进程退出；续跑会复用通过检查的完整断点，未完成 epoch 可能重做。小检查和正式输出不得共用目录。仅改变输出目录不会自动完成旧电脑断点迁移。

查看新电脑进度示例：

```powershell
Get-Content -LiteralPath '.\runs\handoff_i10_formal_001\status.json' -Encoding UTF8
Get-Content -LiteralPath '.\runs\handoff_i10_formal_001\batches\02_D7_I10\status.json' -Encoding UTF8
nvidia-smi
```

LightGBM 使用 CPU；此阶段 GPU 空闲不表示任务停止。

## 迁移缓存、断点：现有入口的实际限制

本包支持同一目录和配置下的 `-Resume`，**尚未实现任意跨电脑路径的自动迁移或断点重定位**。

旧队列配置、来源清单和模型/缓存身份包含绝对路径、环境版本和实现哈希；新电脑的 `team_remaining.py` 还使用便携历史参考适配器，与原电脑的队列配置不完全相同。因此，不要把原电脑队列复制到任意新目录后直接 `-Resume`，也不要批量替换 JSON 路径、删配置或伪造完成标记以绕过检查。

默认可用方式是上面的“只补未完成批次、在新目录正式训练”。如需真正保住原电脑 I10 的未完成断点和缓存，接手 Codex 应先检查配置/缓存身份并做可审查的迁移适配，在副本上验证恢复行为后使用；本文不声称这项迁移已经完成。Windows 的 `.venv-grin` 也不通过复制代替新环境安装。

## 完成后交付什么、如何汇总十方向

每批正式完成后，保留整个正式输出目录，以及它引用的缓存、模型和预测；不要只交一个分数或 `COMPLETE.json`。至少核对：

- 子批次 `COMPLETE.json`、`artifact_manifest.json`、`integrity_check.json`、`run_config.json`、`frozen_models.json` 和来源清单。
- `holdout_comparison_methods.csv`、`holdout_metrics_by_panel_regime.csv`、`holdout_daily_errors.csv`、`paired_calendar_bootstrap.csv`、`holdout_preservation_checks.json`。
- 全部 regime、缺口长度、逐日、通道、孤立节点诊断及训练曲线。按实际生成文件交付，不重命名成虚构的十方向总表。
- 本包会自动向本机代码目录 `实验日志.txt` 追加已完成正式批次记录；核对 `log_update.json`，把新增日志交回原电脑，避免重复记账。小检查结果不进入正式成绩或正式日志。

**当前入口没有“把两个独立批次输出和外部 I210 自动拼成十路报告”的命令。** `-SummaryOnly` 要求原本就是三批完整队列，不能对上述单批输出使用它来假装十路已完成。

最终汇总需要五个已完成实验来源：旧四方向的两个实验、原电脑 I210、本次 I10、本次 D12_I405，共十个方向。先把新电脑产物交回原电脑，由 Codex 使用原评分聚合函数、验证每个来源的完成证明和日期/regime/目标数/回退记录，并适配跨路径来源校验，形成明确标注多机来源的汇总。这一步尚待实现和验收，不是现在 README 已完成的功能。

最终必须给用户：

1. 五种共同方法 T0、T1、Current_fusion、A_dynamic、A_dynamic_fixed 的十方向／五家族总分，以及相对 `0.936910312032` 的差值。
2. 各方向、regime、缺口长度和逐日改善是否稳定，孤立节点回退是否保持不变。
3. 已更新的正式实验日志和完整性检查；随后再决定下一步训练，不自行扩大模型搜索。

单方向或单家族分数不能直接与十方向历史总分比较；本地留出分数也不等于 Kaggle 隐藏测试成绩。

## 若新电脑实际是实验室 Linux 服务器

上面的 PowerShell 安装/启动脚本针对 Windows＋NVIDIA。Linux 上可参考 `team_remaining.py` 的 Python 参数，但本次没有交付经过服务器验证的 Linux 环境或 Slurm 脚本。按实验室指南，训练、推理、评测等重任务必须通过 Slurm；先分配 debug GPU 做小检查，通过后用 `sbatch` 正式运行，不能在普通 SSH 登录终端直接训练。需要另行核对分区、GPU 类型、环境/镜像、存储挂载和实际权限。

## 可直接交给新电脑 Codex 的指令

> 请先读 README_HANDOFF.md 和 README.md，按原完整方案接手，不采用精简方案。先核对两台电脑实时进度与分工，保留所有已完成结果。默认只补未完成的 I10 东西、D12_I405 南北，每方向训练 T1、A_dynamic、T0，保留 LightGBM、原融合和新融合；不要重跑已完成的 I210 和旧四方向。先验证环境和 GPU 小检查，再正式跑。现有入口不支持任意跨路径断点迁移或独立批次直接十路汇总，遇到这些需求先做可验证适配，不绕过哈希与配置检查。完成正式结果后核对日志、完整性和孤立回退，把完整产物交回原电脑用于十路汇总。
