# 验证数据性质更正

本次重新核对原始发布包 `config/synthetic_release_v1.json`：schema 为 `trafficflowbench_synthetic_release_v1`，`traffic_policy.historical_source_rows_in_release` 为 false；Task4 输入文件明确叫 `synthetic_link_counts.csv` 和 `synthetic_weak_prior.csv`。网络元数据也声明未使用历史交通观测行。

因此，历轮报告/聊天里把比赛输入留出检验称为“真实计数验证”“real count CV”的说法不准确，应更正为**比赛公开合成计数留出验证**。旧文件名与结果保留以保证复现；这项更正改变的是证据性质的解释，不改变当时实际计算的数值。它不能证明模型在真实道路实测计数上泛化。

先前单独下载的 Caltrans/PeMS 外部数据，以及本轮 Census LODES、FHWA NHTS 外部数据，属于另一类来源；没有把它们与合成比赛输入混为同一份观测。LODES 是经过统计处理的居住地—工作地就业关联，NHTS 是出行调查，并非竞赛路径标签。

另一个重要限制：发布的 Z0、Z1 等区域没有已验证的 Census/TAZ/经纬度对应关系。每个 OD 在此包里对应一条候选路径。外部地理 OD 不能按区域编号强行贴到比赛路径。路径序列中的部分链接还缺少长度元数据，不能把部分长度之和称为完整行程距离。

v17 的“完整长度”仅指公开 `path_set.link_seq` 中每个路段均在长度元数据里；不代表真实出行全程已知，incidence 中的虚拟 RON/ROFF 连接也没有可核实实测长度。仅在该路径子集上测试外部聚合距离先验，既不补造地理映射，也不补造缺失长度。它仍是迁移假设，不能替代真实路径验证。实际 S_od、S_dev、S_attr 和实际综合分依然未知。
