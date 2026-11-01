# Task2：完整时距监督与Soft-IoU，Overall代理IoU 0.826387575

本版为另存研究候选，原[0.82245版本](../best_overall_082245_20261031/README.md)、[0.82186版本](../best_overall_082186_20261031/README.md)及最好Onset均保留。可靠公开观测代理指标，非官方隐藏S_queue；二月多次用于开发，非新的独立盲测。

|相同二月窗口|Onset|Ongoing|Overall|Ongoing FP/FN|
|---|---:|---:|---:|---:|
|新367(183+184)|0.776032314|0.876742837|0.826387575|690/423|
|旧410(200+210)|0.738098750|0.803951086|0.771024918|517/1026|

新口径相对0.82245：FP739→690，FN412→423；ongoing增益区间：[-0.001137921020346633, 0.023692641421475234]。点估计领先不代表稳定性已确认。

## 固定训练和月份规则

- Onset沿用原early_full9头和门槛[.5,.5,.5,.55,.55,.55]，参考截止2030-11-30，不修改。
- Ongoing在800树上继续100树、lr.02、63叶/深度7/最小叶60/L2=20；24份900树权重包含初始化800树，不需要额外加载旧Ongoing权重。固定三种子20261030/31/20261101。
- 原274维因果特征、同8143拟合窗口和节点IPW；改用每节点全部已知5–30分钟时距。未知未来不进损失，未来标签仅用于拟合目标和权重，不是查询特征。主线masked-T、匝道T-5。
- 相同训练数据、相同逆密度窗口权重的BCE对照，及BCE+lambda窗口SoftIoU(.25/.75)。本版lambda=0.75。SoftIoU采用数值差分核验的解析梯度和正值曲率近似。
- 开发权重6—9月，十月固定.25选两组SoftIoU融合配置；十一月三种子报告、不重选。BCE固定对照在一月和二月前追加、最终三种子共同评估。R127训练过开发日期，锚定开发并非完全样本外。
- 最终仅6—11月拟合。本轮以更高IoU为主目标，一月仅选择全局IoU最高门槛0.375，记录FP/FN约束可行性而不阻止主门槛。配置、权重、目标源代码和门槛在本轮二月评分前哈希冻结。二月没有调参。
- 模型原始输出统一通过sigmoid，再三种子平均并与R127等权log-odds融合。两套窗口同一模型，不按道路或口径切换候选。Overall完整16组时等于(Onset+Ongoing)/2，不以全单元格IoU替代。

主门槛0.375在一月对应FP773/FN573，略超原FN保护条件（1.05×544=571.2）；受限备选门槛0.35仅记录，未用于本轮二月预测。日志stdout的feasible表示存在某个受限备选；主门槛的实际状态以January_results.json的feasible=false为准，详见reports/logging_note.json。该显示问题不改变权重、门槛或分数。

## 原始观测复评

保留本目录、best_overall_081796_20261030及r127_reproduction_20261027相对位置，安装requirements.txt与R127依赖，自行获取授权Kaggle原数据。下载[R127完整Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)，其ZIP SHA256为c90229ea2df8215ffe18c52de910b1c32735d569257e36b076b0c3e9e3f07649。

```powershell
python evaluate.py --package D:/models/ConditionalPosition-R127 --data-root D:/data/kaggle_public --work D:/work/task2-newbest --population both --causality
```

入口核验发布和原父链文件、原始数组哈希；从历史观测复算777个窗口，逐道路×条件核对到1e-12，并检查改变未来标签/掩码/速度不改变预测。验收记录见acceptance/。原始数据不入库。

已有同顺序特征NPZ(X[N,274],panel[N],parent[N,6])时：

```powershell
python code/predict_ongoing_roads.py --models . --features D:/work/features.npz --candidate j075_odds50 --output D:/work/prediction.npz
```

特征顺序见feature_names.json；输出概率、固定门槛与二值预测。reports/包含所有候选和固定对照，不按二月重配参数。training_archive/是实验代码，完整重训仍需原始数据、初始化模型与训练缓存，不声称一键从零重训已验证。manifest.json为所有交付文件SHA256。
