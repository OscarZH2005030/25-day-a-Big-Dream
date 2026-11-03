## Task1 八道路控制组本地小幅提升（2026-11-02，研究候选）

[权重、便携推理与完整限制说明](experiments/local_control_20261102/README.md)：同口径已开发55日S_state从0.942828632到0.942948671，一月、二月均小幅改善；八道路移动入口逐一重算通过。不是官方总分，未提交Kaggle，不宣称0.95。原版保持不变。

## Task 2 当前最高新口径代理Overall点估计：0.828148579（2026-11-01）

[固定模型、一月门槛配置与原始数据复评入口](experiments/task2/best_overall_082815_smooth_20261101/README.md)

**最好Onset early_full与全部模型权重不变**，仅将 ongoing 门槛改为一月选出的平滑时距门槛。二月相同新367正例窗口：Overall **0.828148579**，Onset **0.776032314**，Ongoing **0.880264843**。相对[保留的0.82723版](experiments/task2/best_overall_082723_horizon_20261031/README.md)，ongoing误报681→595，但漏报417→470。主窗口ongoing增益95%区间[-0.000748,0.004529]仍跨零。相同模型在旧410窗口ongoing **0.801366893**、Overall **0.769732821**，低于原0.82723版的0.806011104、0.772054927；旧窗口退步，不宣称全面或稳定提升。旧版全部保留，不按窗口切换模型。

仅用一月选择门槛的两个参数，模型、代码和门槛哈希冻结后在二月比较，未用二月调参。777个原始窗口复算、32项未来标签/掩码/速度扰动检查通过。分数是可靠观测代理，不是官方S_queue；二月反复用于研究，非独立盲测。新增轨迹特征模型本轮训练未超过原版，结果作为失败对照保留。

[原0.82723模型与完整新增残差权重](experiments/task2/best_overall_082723_horizon_20261031/README.md) · [0.82639父模型](experiments/task2/best_overall_082639_jaccard_20261031/README.md) · [R127完整Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)。本次权重完全不变，新增配置依赖以上原目录。

[原月度二月64个Ongoing窗口的同口径核对](experiments/task2/r127_same_window_audit_20261031/README.md)：R127 **0.894560498**，原0.82639模型 **0.911464403**，0.82723分时距模型 **0.913578007**。这里的0.91358是原门槛在另一64窗口上的ongoing，不能作为本版Overall或主184窗口成绩。

## Task 2：12—2月去重评估的保真影子候选（2026-10-31）

[权重、独立推理入口、去重分数重算与保真审计](experiments/task2/shadow_pending_dedup_20261031/README.md)。固定1479个评分窗口上，R127为 **0.788043545011**，pending专家影子候选为 **0.790893192093**；6/6、17/17关键真阳性保留，各检查月份无新增已知真阳性损失。原始两版2月23个重复窗口只计一次。

这是反复开发数据上的public-speed proxy，不是官方隐藏分数或独立验证；**不替换默认模型，也不与上面的2月单独0.817962口径直接排名**。提供约15MB的新增运行包、父链Release依赖和8道路可迁移验收；没有上传原始竞赛数据。

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
