# Task 2：LightGBM 队列预测 Baseline（初版）

> **状态：可复现的本地训练结果与候选提交文件已打包。** 本文分数来自公开训练集内部的代理标签验证，不是 Kaggle 官方评分；validation/private 文件是候选预测，尚未作为 Kaggle Submission 提交。

## 1. 任务与交付摘要

| 项目 | 本次实现 | 结果/说明 |
|---|---|---|
| 竞赛任务 | Task 2：队列发生与传播预测 | 按官方提供的窗口历史和模板生成目标时刻的队列二分类预测 |
| 负责范围 | Oscar 的 Task 2 模型训练 | 单独训练 onset 与 ongoing 两类状态模型 |
| Baseline | LightGBM（二分类） | 两个独立模型，概率阈值固定为 0.5 |
| 输入窗口 | 最近 60 分钟历史 | 12 个 5 分钟滞后点，叠加道路、时空与邻接路段信息 |
| 训练数据 | 80 个训练窗口 | 共 87,000 个目标单元；训练样本来自官方公开 train split |
| 内部留出 | 19 个完整窗口 | 每个 panel/condition 按日期留出最新日期的全部窗口：onset 9 个，ongoing 10 个 |
| 主指标 | 内部代理 IoU | onset/ongoing 等权平均：LightGBM **0.500764**，Persistence **0.330162** |
| 预测产物 | validation 与 private 候选 CSV | 各 87,000 行，按官方 sample submission 的键输出；并附概率 Parquet |
| 官方提交 | 尚未提交 | 当前结果不能称为 Kaggle leaderboard 分数 |

## 2. 数据、标签与验证设计

| 环节 | 做法 | 含义与注意事项 |
|---|---|---|
| 数据范围 | 读取官方 Task 2 公开 train/validation/private 分区及各 panel 的窗口文件 | 训练标签只由公开 train 的未遮挡观测构造 |
| 代理标签 | 对有效观测，以 `speed <= 0.60 × free_speed` 标记拥堵/队列 | 官方 `queue_true` 是隐藏潜在状态；该规则只是可训练代理，不等同官方真值 |
| 可评估目标 | 仅在符合条件、可由观测判断的目标单元参与代理训练/评估 | 不把无效或不可判断样本当作可靠真值 |
| 时间切分 | 按 panel 与 condition 分组，将日期最新的完整窗口留出 | 避免同一窗口的目标行拆到训练和验证两边；窗口数量仍较少 |
| 基线对照 | 在相同内部留出键上对比 LightGBM 与 Persistence | 两种方法使用同一代理标签和 IoU 口径 |
| 最终拟合 | 留出评估完成后，以所有 80 个 train 窗口训练提交候选模型 | 留出分数是开发评估；最终模型包含这些留出窗口重新拟合，不能再称独立测试分数 |

## 3. 模型与特征

| 模块 | 实现内容 |
|---|---|
| 模型结构 | 两个 LightGBM binary classifier：`queue_onset`、`queue_ongoing` 各一个 |
| 预测输出 | 对模板每个目标键输出 `queue_pred`；概率另存为 Parquet，标签由 0.5 阈值产生 |
| 历史交通特征 | 12 个滞后点的速度比、流量利用率、占有率、历史队列状态及有效观测标记 |
| 汇总与趋势特征 | 3/6/12 步速度均值与最小值、流量均值、队列计数、有效样本数，以及 15/30/60 分钟速度变化 |
| 空间特征 | 上下游相邻主线的最近速度比、3/6 步均速、当前队列与近期队列计数 |
| 道路属性 | 车道数、路段长度、自由流速度、容量、临界密度、堵塞密度、拓扑顺序、匝道连接标记 |
| 时间与预测位置 | 预测 horizon、onset/ongoing 编码、起报小时/星期及小时正余弦编码 |
| 特征总数 | 106 |

## 4. LightGBM 训练配置

| 参数 | 值 |
|---|---:|
| objective | `binary` |
| learning_rate | 0.035 |
| num_leaves | 15 |
| min_data_in_leaf | 80 |
| feature_fraction | 0.9 |
| lambda_l2 | 2.0 |
| boosting rounds | 250 |
| threshold | 0.5（固定，未根据留出集调阈值） |
| seed | 2026 |
| threads | 4 |
| deterministic | true |

## 5. 内部代理验证结果

| 方法 | 条件 | IoU | 解读 |
|---|---|---:|---|
| LightGBM | ongoing | 0.744914 | 对持续队列识别较好，明显高于持续状态 Persistence |
| Persistence | ongoing | 0.549212 | 延续最近可见队列状态的对照结果 |
| LightGBM | onset | 0.256614 | 新发队列识别仍弱，是当前主要短板 |
| Persistence | onset | 0.111111 | Persistence 对新发状态本身不敏感 |
| LightGBM | onset/ongoing 等权均值 | **0.500764** | 内部 train 代理评估，不是官方分数 |
| Persistence | onset/ongoing 等权均值 | **0.330162** | 与 LightGBM 同切分、同口径的参考值 |

**结果说明：** LightGBM 在这份内部代理评估中整体优于 Persistence，提升主要来自 ongoing；onset 绝对 IoU 仍低。由于标签是速度阈值代理且留出窗口有限，这只能说明值得继续验证，不能据此断言官方隐藏状态预测已有相同表现。

## 6. 仓库交付文件

| 路径 | 内容 | 用途 |
|---|---|---|
| `experiments/task2/train_lgbm_task2.py` | 训练、验证和候选预测脚本 | 在仓库根目录、安装依赖并准备 `data/kaggle_public` 后运行 |
| `artifacts/task2_lgbm_candidate/lightgbm_condition_onset.txt` | onset LightGBM 权重 | 模型复用/推理 |
| `artifacts/task2_lgbm_candidate/lightgbm_condition_ongoing.txt` | ongoing LightGBM 权重 | 模型复用/推理 |
| `artifacts/task2_lgbm_candidate/queue_lgbm_validation.csv` | validation 模板候选标签 | Kaggle validation 候选预测 |
| `artifacts/task2_lgbm_candidate/queue_lgbm_private.csv` | private 模板候选标签 | Kaggle private 候选预测；不得用其反馈调参 |
| `artifacts/task2_lgbm_candidate/*_probabilities.parquet` | 两份候选的逐键预测概率 | 阈值调整、错误分析与后续集成 |
| `artifacts/task2_lgbm_candidate/internal_train_proxy_summary.csv` | 分模型/分条件 IoU 汇总 | 快速查看内部代理结果 |
| `artifacts/task2_lgbm_candidate/internal_train_proxy_by_window.csv` | 逐留出窗口代理结果 | 分窗口错误分析 |
| `artifacts/task2_lgbm_candidate/feature_importance.csv` | 特征重要性 | 检查模型主要使用的特征 |
| `artifacts/task2_lgbm_candidate/provenance.json` | 模型、切分、特征和评估限制元数据 | 追溯本次产物来源 |
| `docs/results/task2_internal_train_proxy_summary.csv` | 轻量结果副本 | 文档/绘图引用 |
| `docs/results/task2_internal_train_proxy_by_window.csv` | 轻量逐窗口结果副本 | 文档/绘图引用 |

## 7. 产物检查与可复现性

| 检查项 | 结果 |
|---|---|
| validation 候选行数 | 87,000 |
| private 候选行数 | 87,000 |
| 模板键匹配 | 两份 CSV 均按各自官方模板键输出 |
| 重复预测键 | 0 |
| `queue_pred` 取值 | 仅 0/1 |
| 概率输出 | validation/private 各一份 Parquet |
| 模型产物 | onset 与 ongoing 两个 LightGBM 原生模型文件 |
| 提交状态 | 仅打包为候选文件；未上传 Kaggle、未获得官方分数 |

复跑命令：

```bash
python experiments/task2/train_lgbm_task2.py
```

脚本默认读取 `data/kaggle_public` 下的 Task 2 数据，写入被 Git 忽略的 `outputs/task2_lgbm_candidate/`。本次提交到仓库的 `artifacts/task2_lgbm_candidate/` 是供组员检查和复用的快照。若数据文件或依赖版本不同，复跑结果可能变化；应记录环境与代码版本。

## 8. 当前问题与后续改进

| 问题 | 对结果的影响 | 建议下一步 |
|---|---|---|
| 代理标签不是隐藏的 `queue_true` | 内部 IoU 与官方评分目标可能不一致 | 以 Kaggle 提交结果作为外部验证；报告中继续明确标注代理性质 |
| onset IoU 仅 0.256614 | 新发队列预测能力不足，整体均值被拉低 | 检查 onset 正负样本比例、分 horizon 召回/精确率；仅用训练内部时间切分试阈值与代价权重 |
| 训练窗口只有 80 个，留出 19 个 | 窗口层面的估计不稳定，换日期/路段可能波动 | 增加按日期滚动验证，汇报不同切分的均值和方差 |
| 0.5 阈值目前固定 | 不一定适合官方指标或类别比例 | 在 train 内部交叉验证选择阈值，并保留未调阈值结果作对照 |
| 当前主要依赖浅层历史与邻接统计 | 远距离传播/队列动力学可能建模不足 | 在统一切分和同一指标下，与 GRIN/Transformer 或物理约束模型比较 |
| private 候选已生成但没有真实标签 | 无法离线计算 private 正式指标 | 只生成一次最终文件并按竞赛规则提交，不从 private 结果反向调参 |

## 9. 结论

这次完成了 Task 2 的第一版模型训练与内部代理验证：建立了 onset/ongoing 两个 LightGBM 分类器，训练后生成 validation/private 候选预测及复核材料。内部相同留出窗口上，LightGBM 的等权代理 IoU 高于 Persistence（0.500764 对 0.330162），但 onset 表现偏弱，而且代理标签与官方隐藏真值并不相同。当前应把它定位为**可复现 baseline 和提交候选**；正式优劣仍需通过 Kaggle 评分及组内统一实验协议确认。
