## Task 2 当前最高新口径代理Overall：0.817962444（2026-10-30）

[固定模型、全部新增权重与原始数据复评入口](experiments/task2/best_overall_081796_20261030/README.md)

采用 **新Onset early_full + 原R127 Ongoing**。二月新口径367个正例窗口：Overall **0.817962444**，Onset **0.776032314**，Ongoing **0.859892575**；原R127 Overall为0.795976209，提升2.20个百分点。同一模型在旧410窗口上Overall为0.758598036。三种子及门槛固定，只用一月选择门槛，二月未调参；近期Ongoing实验未提升新口径，故本版保留R127 Ongoing。

以上是公开带噪观测的代理评分，不是官方S_queue；二月反复用于研究，不能视为独立盲测。不同窗口口径分别报告，不拼接道路或按口径切换模型。新增权重已入库，原R127完整父链沿用[既有Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)。

[原R127完整复现](experiments/task2/r127_reproduction_20261027/README.md) · [原0.79598版本与逐月审计](experiments/task2/r127_reproduction_20261027/reports/current_best_20261029/README.md)

# 25-

## Task 2：数据审计、代理窗口与图时空模型小检查（2026-10-24）

[RTX 4070／Windows 独立环境、代码与运行说明](experiments/task2_queue_20261024/README.md)。支持八方向审计、冻结月份、按日复用缓存、代理标签、persistence／LightGBM 基准与传播延迟图时空 Transformer 的有限小检查。当前没有正式训练入口；不修改 Task 1，代理指标不代表官方隐藏队列成绩。

## 当前最佳模型：Strong v4（2026-10-20）

[完整模型说明、推理入口、实验记录与恢复指南](experiments/task1_strong_v4/README.md)。

三个重点方向采用 4,386 万参数图/SSM/全天注意力主干，其他方向沿用 v3；完整依赖包含 27 份已选用权重。相同 55 天本地留出集的 S_state 为 **0.939898101**，每车道流量 RMSE 为 **34.320188**，速度 RMSE 为 **1.669400 km/h**。该本地留出集已用于多轮开发，尚无 Kaggle 隐藏测试分数。

完整模型包与已校验提交通过 [GitHub Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/ieee-strong-v4-20261020) 分发；仓库保存源代码、配置、评估结果和 SHA256 文件清单。新目录的可迁移推理已在十方向检查通过。

## 最新训练实验：T1／动态缺失六方向队列（2026-10-21）

[Windows＋NVIDIA 新电脑运行指南与完整代码](experiments/task1_t1_dynamic_20261021/README.md)。本次训练剩余六方向的 T1、动态缺失与 T0，共18个神经模型；沿用固定划分、预算、评分和孤立节点回退。随包提供经原始来源清单校验的已完成四方向参考表，新六方向完成后自动生成十方向对比。该队列仍在实验中，历史目标线为0.936910312032。

## Task 1：LightGBM、GRIN 与长缺口组合

[代码、正式实验结果与小组运行指南](experiments/task1_january_grin/README.md)。包含十方向纯 LightGBM baseline，以及 I5 / D7 的 GRIN 和固定长缺口切换对比。


## Task1：0.9428完整复现

[从零训练、全55日评估、Task1预测与完整Kaggle合并流程](experiments/task1/best_road_combination_20261026/reproduction/README_完整复现.md)。发布权重重算S_state=0.94282863197，6,636,705个目标；验证/私有Task1预测6,735,795行。属于已查看过的本地选模验证，不是Kaggle总分。
