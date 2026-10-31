# Task2：overall代理IoU 0.817962444（2026-10-30）

固定组合为 **新Onset `early_full` + 原R127 Ongoing**。本目录包含新增Onset的全部9个LightGBM头、8份历史参考快照、位置基、固定配置、推理代码与实验记录。R127的完整父链权重沿用既有Release，无需重复训练。

| 固定二月口径 | 正例窗口 | Onset | Ongoing | Overall | 原R127 Overall |
|---|---:|---:|---:|---:|---:|
| 新口径 | 367（183+184） | 0.776032313657 | 0.859892575084 | **0.817962444371** | 0.795976208550 |
| 旧口径 | 410（200+210） | 0.738098749884 | 0.779097322813 | **0.758598036349** | 0.754599304093 |

两套口径使用同一模型，不拼接道路或切换候选。Overall按“窗口→道路×条件→道路→四类道路”等权汇总；完整16组时等于 `(Onset + Ongoing)/2`，不是按367个窗口或单元格数直接平均。

这些分数是可靠公开观测构造的**代理IoU**，不是官方`S_queue`。`queue_true`未公开；二月此前已多次用于研究，不是新的独立盲测。新口径最高点估计用于标记本次发布，不代表误报、漏报或所有道路均改善。

## 固定配置与代价

- 三种子：20261030、20261031、20261101，平均概率。
- Stage1沿用上一轮996维位置/时段先验与相对常态事件头。
- Stage2：前三步使用独立条件排队头，后三步沿用起止区间头。
- Onset解码：连续区间最大效用，门槛`[.50,.50,.50,.55,.55,.55]`；Ongoing保留R127 `.25`。
- 6—9月开发拟合，十月筛选，十一月比较，6—11月最终拟合，**只用一月选择上述门槛**。模型与门槛在本次二月评估前哈希冻结，二月未调参。
- 新口径Onset：FP267/FN528；R127为FP327/FN467。旧口径Onset：FP387/FN577；R127为FP374/FN569。FP/FN是已知单元格数。
- 后续Ongoing实验没有提升新口径，因此本版继续采用原R127 Ongoing。

## 获取依赖权重

安装本目录`requirements.txt`，以及[R127原环境要求](../r127_reproduction_20261027/requirements.txt)。原始数据需自行从Kaggle授权获取，本仓库不包含原始数据或本机运行环境。

下载并解压[R127完整模型Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)中的`ConditionalPosition-R127.zip`。其SHA256为：

```text
c90229ea2df8215ffe18c52de910b1c32735d569257e36b076b0c3e9e3f07649
```

## 从公开原始观测复算两套二月成绩

在本目录执行，路径替换为自己的数据与解压目录：

```powershell
python -m pip install -r requirements.txt
python evaluate.py --package "D:/models/ConditionalPosition-R127" --data-root "D:/data/kaggle_public" --work "D:/work/task2-081796" --population both --causality
```

程序验证本目录manifest、R127的1214个原文件和重建原始数组哈希；直接从历史观测构造996维特征，重新预测两个条件。Onset采用向量门槛的区间解码；Ongoing直接调用未改动的R127。输出`metrics.json`和`verification.json`，逐道路×条件与已保存结果比较到`1e-12`。`--causality`检查改变当前窗口未来标签与掩码不会影响预测。

全局常态及风险参考固定在2030-11-30，推理时不读取一月或二月标签。主线历史截止masked-T，匝道历史截止T−5。缺失未来单元格只在评分中掩码。

## 已有996维因果特征的独立Onset推理

```powershell
python code/predict_early_onset.py --models . --features "D:/work/causal_features_996.npy" --candidate early_full --output "D:/work/onset_prediction.npz"
```

特征顺序为940维基础特征、8维历史先验、48维相对常态特征。名称见`feature_names.json`；输出概率`[N,6]`和区间二值预测。本版只发布`early_full`所需权重；`January_results.json`保留了原始校准对照记录，不表示其他候选权重也包含在此目录。

`training_archive/`保存原实验训练代码和路径布局记录，完整训练仍依赖原始数据与中间训练缓存，不是已验证的一键从零重训入口。本次验收目标是冻结模型推理和代理评分复现。R127自身的精确重训流程见其独立复现目录。

本目录所有文件的SHA256见`manifest.json`，完整复评验收记录见`acceptance/`。
