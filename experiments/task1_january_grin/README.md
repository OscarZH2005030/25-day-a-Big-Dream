# Task 1：LightGBM、GRIN 与长缺口组合

这是本小组的六月至十二月训练、一月开发实验快照。代码、正式结果表、配置、训练曲线、实验日志和分工表一并共享。原始比赛数据、虚拟环境、缓存、逐目标预测和模型权重保留在本地，需要重跑时按 [小组复现指南](TEAM_REPRODUCE.md) 生成。

## 先看哪里

- [结果总表 CSV](reports/summary.csv)：十方向纯 LightGBM 及两方向组合结果。
- [实验日志](实验日志.txt)：按实验记录做过的尝试。
- [十方向 baseline 对比](reports/all_baseline_resume_20261017_011520_258735/comparison_overall.csv)；[分方向结果](reports/all_baseline_resume_20261017_011520_258735/comparison_by_panel.csv)。
- [I5 三种方法对比](reports/grin_hybrid_i5_20261017_142725/comparison_hybrid.csv)。
- [D7 三种方法对比](reports/grin_hybrid_d7_20261017_143642/comparison_hybrid.csv)。
- [十方向 baseline Excel](Task1_Baseline_全部道路结果_20261017.xlsx)；[三人任务和 DDL](TrafficFlowBench_三人任务分工与DDL.xlsx)。

## 当前结果

S_state 越高越好。两方向得分不能当作十方向总体成绩。

| 方法 | 十方向 / 五走廊家族 | D12_I5_N | D7_I405_N |
|---|---:|---:|---:|
| 多跳残差 LightGBM | 0.921714 | 见原始结果表 | 见原始结果表 |
| 多跳直接 LightGBM | 0.921431 | 见原始结果表 | 见原始结果表 |
| LightGBM 两模型等权 | 0.922768 | 见原始结果表 | 见原始结果表 |
| 冻结 LightGBM baseline | **0.922849** | 0.928029 | 0.915051 |
| GRIN | 尚未运行全部方向 | 0.917990 | 0.900606 |
| baseline＋GRIN 长缺口切换 | 尚未运行全部方向 | **0.929411** | **0.915659** |

冻结 LightGBM baseline：双侧可见速度观测间隔 ≤60 分钟时，残差与直接预测等权；>60 分钟或缺一侧时，使用直接预测。它只包含 LightGBM，不包含 GRIN。

长缺口组合：仅当双侧可见速度观测间隔有限且 >60 分钟时用 GRIN，其余沿用冻结 baseline。缺一侧仍用 baseline；两个通道共用规则，不用真值选择模型。

该规则在 I5 开发分析后固定，在 D7 保持不变。两方向总分都提升，收益主要来自 R3。D7 的 R1 略退步，逐日胜负为 16/15；I5 为 21/10，因此还不能声称逐日、所有 regime 均稳定提升。D7 拥堵子集也未改善。下一步集中改善 GRIN 训练稳定性，保持 baseline 为对照。

## 实验范围与评分

- 拟合：2030-06-01 至 2030-12-31；开发：2031-01-01 至 2031-01-31；二月未使用。这是数据中的合成日历。
- 每方向每天抽取 1000 个原缺失目标训练，seed=20261016；开发用完整目标集合。GRIN 正式运行核对了与 baseline 相同的训练目标键。
- `task1/metrics.py`：速度 RMSE /25，流量先除车道数再算 RMSE /600；分数权重 0.54 / 0.46，各通道分数下限为零；依次按 regime、方向、走廊家族等权汇总。总表 RMSE 也按该层级平均，不是把全部目标混算。
- GRIN 为独立实现的核心结构，未声明与原论文实现完全一致。参考：[GRIN 原始代码](https://github.com/Graph-Machine-Learning-Group/grin)。
- I5、D7 的 GRIN 分别在各自方向训练，不是跨走廊迁移同一组权重；本次均为单随机种子。
- 仓库其他分支中已有的 `task1_time80_20` 等实验使用其他划分，需对齐数据协议后才可比较分数。本目录不覆盖这些实验。

## 代码和历史记录

`task1/` 包含数据审计、评分、LightGBM、特征/目标消融、融合、缓存和 `grin/`；`tests/` 为现有校验；`run_*.ps1` 和 `setup_*.ps1` 为 Windows 入口。依赖基于 Python 3.14 的原实验锁定版本，GRIN 环境为 PyTorch 2.10.0+cu128。

历史说明按阶段保留：[最初 baseline](README_TASK1.md)、[特征组](README_TASK1_EXPERIMENTS.md)、[残差与直接预测](README_TASK1_TARGETS.md)、[LightGBM 融合](README_TASK1_FUSIONS.md)、[第二方向验证](README_TASK1_CORRIDOR.md)、[GRIN](README_TASK1_GRIN.md)、[长缺口组合](README_TASK1_GRIN_HYBRID.md)。其中原电脑路径和 `runs/` 默认值是历史记录，换电脑请优先按 [小组复现指南](TEAM_REPRODUCE.md) 操作。

`reports/` 按原 run 名保存正式小型结果，`export_manifest.json` 记录原文件 SHA256，未重算或改写历史分数。`dev_20261016_160422` 是早期完整 baseline 结果，当时尚无完成标记；`grin_hybrid_i5_20261017_142725` 是复用已有预测的组合复核，未训练、未重复写日志。中断和 smoke 目录没有作为正式结果上传。

本目录的 `reports/` 供阅读和审计，不是可执行的 `runs/`。历史 JSON 中的本机绝对路径仅用于溯源；没有上传缓存、预测和权重，不能直接运行历史默认组合命令。
