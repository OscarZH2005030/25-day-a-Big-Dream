# Task3 状态候选与 Task4 需求结构研究（2026-11-02）

本目录记录可复核的本地研究，**没有新 Kaggle 提交，也没有证明官方总分或完整 Task3 提升**。用户要求保留剩余两次提交机会。官方已测基线 0.81135 来自用户评分回执；继续保留该方案，Task4 不替换。

## 状态预测：可运行的两套独立候选

下表均为公开 train 的二月开发评价，十方向/三个原始 mask regime 的宏平均。二月已反复用于研究，不能称独立盲测。这里的积累变化 RMSE 只比较相邻目标的真实积累变化，不是官方 LWR。

| 方案 | S_state | 速度 RMSE | 每车道流量 RMSE | 积累变化 RMSE |
|---|---:|---:|---:|---:|
| 发布基础模型 | 0.94340048 | 1.49124334 | 31.81129895 | 1.95441048 |
| 夜间 context | 0.94534737 | 1.43560439 | 30.83944878 | 1.84155729 |
| 多月训练 | 0.94567387 | 1.42646269 | 30.67113541 | 1.82543320 |
| 严重缺失增强混合 | **0.94581962** | **1.42179113** | **30.61263506** | **1.81864846** |

- `task3/packages/multimonth/`：十二月14—31日、一月5—20日训练，按月份和 regime 平衡；一月21—31日选路线/强度，未选路线保留 context。
- `task3/packages/severe_mixture/`：同一训练期加入站点和长块缺失监督；原始/站点/长块损失权重为50%/25%/25%。一月原始和缺失条件选保守混合，全部道路冻结后才评价新二月掩码。
- 每套包都有实际推理代码、独立权重、来源和哈希。模型权重和推理源码与本地校验版本相同，发布元数据仅去除本机绝对路径。
- 两套包各自完整20份 validation/private 状态部分，共6,735,795行，已核对模板键、有限非负值和独立CSV回读；这里只保存验收哈希，避免提交巨量状态CSV。

**保留的局限：** 严重缺失候选在新严重缺失测试的平均状态和积累指标均改善；在三个旧轻度掩码集合中状态均值改善，但 point10 的平均积累RMSE从 context 的1.59904014变为1.61038603，仍有退步。个别道路可变差，不应按二月结果重新挑路。站点遮挡仅隐藏合格可观测单元，其他原始信号可能保留。

固定三种子的后续平均仅小幅改善（开发状态0.94582071、积累RMSE1.81795372），全部种子结果保留；暂不把它包装成更高官方成绩。时间平滑、直接积累回归和时间缺口门控未通过一月约束，负结果在 `results/`。

### 独立推理

先使用原发布基础模型生成 keyed Task1 状态CSV，再在同一份原始发布数据上应用校准。基础模型来源提交为 `chengweiliu669-pixel/25-day-a-Big-Dream-submission@e254d55acdb94261d03d2ea10a79c983f1e15d55`，须按该仓库发布权重复现，不以其他基础预测替换后仍声称同分。

```sh
pip install -r requirements.txt
python task3/packages/severe_mixture/apply_residual.py --data-root /path/to/kaggle_public --state-csv /path/to/baseline_state.csv --panel D12_I405_N --split validation --output /path/to/candidate_state.csv
```

离线 Task1 特征可能使用同日未来观测，不适用于在线 Task2。保留输入基线，输出另存；不会合并完整提交或访问 Kaggle。

公开 train 上重算状态诊断：

```sh
python task3/evaluate_state.py --release-root /path/to/kaggle_public --package task3/packages/severe_mixture --baseline-state /path/to/february_baseline.csv --start 2031-02-01 --end 2031-02-28 --output /path/to/evaluation
```

完整复算须提供完整原模板目标。`baseline-state` 也支持多个 parquet。推理结束后才读取公开 train 真值。

## Task4：重点是恢复 OD 配对，而非继续逼近 S_link=1

官方：`S_ODME=.45*S_od+.25*S_link+.15*S_dev+.15*S_attr`。S_dev比较偏离弱先验的幅度与真实幅度，不是越贴近先验越好。

实际发布网络审计得到：

1. 十个方向都为一个OD对对应一条发布路径，主要是在恢复OD矩阵。D12_I405_N有903个OD对，计数矩阵秩42；全部起终点边际已知仍有820个线性不可识别方向。
2. 在30个方向/切分上，严格满足公开计数和非负性的总需求最大/最小比值达18.68—30.89。D12_I405_N validation为25.40倍，两端计数NMAE不超过浮点误差。这是可行性范围，不是合理需求估计。弱先验会排除一部分极端解。
3. 更严格的160个真实网络反例，同时保留基准计数、全部起终点边际、先验L1偏离量。S_link/S_dev/S_attr因此保持相同，但OD向量相对基准的L1差异仍为0.2368—0.6406。它们既不是更好模型，也没有隐藏分数；说明S_od的45%承担核心区分作用。
4. 公开主线UTC15—19均值与Task4聚合计数平均仍有约3.6%—3.9%的NMAE，原始匝道ID也未直接匹配Task4计数。不能未经时间/生成/拓扑核验就把它们加入硬约束。

`task4/frozen_reference_path_flow.csv` 是原提交仓库的冻结预测，**不是隐藏真值**。用于重现上述不变性实验，不能拿它给自己算“真实 S_od”。v20原始实现未在本机找到，未猜造或重建其代码。

### 从原始公开数据独立复算结构审计

```sh
python task4/audit_task4.py --release-root /path/to/kaggle_public --baseline-path-flow task4/frozen_reference_path_flow.csv --output /path/to/task4_audit
```

默认全部十方向、validation/private、每例八个固定随机线性目标；支持 `--panels D12_I405_N --splits validation --objectives 2` 小检查。不需要旧工作区、Kaggle凭据或GPU，不修改任何预测。完整30例需求边界另见 `results/task4_demand_bounds/`。

后续重点：检验弱先验的误差结构以及哪些结构能由公开约束支持，不再把链路拟合的小数点优化当主要进展。人工压力和计数CV均不能证明隐藏路径改善。

### 新进展：先验误差协方差的跨走廊对照

只用十方向 `train` 的公开 `c-A*b` 残差，比较独立链路、加性路径、乘性路径、起终点相关、学习混合五种协方差假设。去除重复计数方程，按走廊家族留一交叉检查；全部拟合冻结后才评价公开validation/private输入。预测负对数似然越低越好，**不是隐藏S_od或Task4得分**。

| 假设 | 留一走廊 NLL/维 | validation NLL/维 | private公开输入 NLL/维 |
|---|---:|---:|---:|
| 固定加性路径 | **-0.755498** | **-1.129545** | **-1.152009** |
| 学习混合 | -0.753551 | -1.127918 | -1.150571 |
| 起终点相关 | -0.470623 | -0.903211 | -0.906847 |
| 乘性路径 | 0.245037 | -0.402988 | -0.346680 |
| 独立链路 | 1.193094 | 0.362677 | 0.319870 |

学习混合在全train拟合时约99.95%权重落在加性路径分量，但其留一走廊结果没有超过固定加性模型。因此目前没有数据依据仅因乘性/KL形式更复杂就认定其更好。**这不是证明真实生成机制为加性噪声**：只有十个训练方向，计数无法识别任意零空间误差，误差分量也可能共线，且公开输入反复曝光。仍未替换冻结Task4预测。

协议、五个留一走廊的逐道路预测似然、所有优化启动及失败状态、协方差相似度和输入指纹在 `results/task4_covariance_audit/`；代码快照在 `archive/daytime/task4_covariance_audit.py`。后续检验弱先验的OD结构能否在不同道路和未用于拟合的先验条目上复用，不能把对弱先验的预测当真实OD验证。

## 证据和复现范围

- `results/`：登记协议、种子、逐道路/日期的聚合结果、数据/模型哈希、失败与导出验收。没有原始竞赛观测或逐单元训练真值。
- `archive/`：原研究脚本快照，记录原工作区/缓存依赖，供审阅训练与选择过程；这些不是通用运行入口。可迁移入口为上述 `task3/` 和 `task4/` 脚本。
- `environment.json`：实际验证环境版本；`SHA256SUMS.txt`：本目录文件指纹。
- 49日重拟合的一月已训练，其一月验证分null；夜间late55的二月已训练，其二月验证分null。绝不能套用其父模型验证成绩。
- 公开拓扑回退LWR对完美预测也触底，未用来选模型或声称官方Task3提高。

官方来源：[评分合同](https://github.com/leigaadik/trafficflowbench/blob/main/docs/SCORING_SPEC.md)、[数据及本地评分限制](https://github.com/leigaadik/trafficflowbench/blob/main/docs/DATA.md)。
