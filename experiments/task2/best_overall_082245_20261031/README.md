# Task2：保留0.82186，新增Overall代理IoU点估计0.822451408

本版在已发布的[0.821855853版本](../best_overall_082186_20261031/README.md)上继续训练Ongoing。**原0.82186与最好Onset均保留**。新模型仅有小幅IoU点估计增益，并增加了漏报，作为另存研究候选发布，不宣称已经确认稳定提升。

| 同一二月新口径 | Ongoing IoU | Overall | Ongoing FP | Ongoing FN |
|---|---:|---:|---:|---:|
| R127 Ongoing + 固定最好Onset | 0.859892575 | 0.817962444 | 922 | 372 |
| 保留0.82186版本 | 0.867679393 | 0.821855853 | 789 | **377** |
| 本版inverse加权候选 | **0.868870502** | **0.822451408** | **739** | 412 |

新口径367个正例窗口（183 Onset、184 Ongoing）的Onset仍为0.776032314。相对0.82186，本版少50个FP、多35个FN；IoU增益0.001191109的按日分块95%区间为[-0.001393315,0.003893852]，仍跨零。

旧410窗口（200 Onset、210 Ongoing）使用完全相同的权重及门槛：Onset0.738098750、Ongoing0.804078019、Overall0.771088384、Ongoing FP548/FN986；0.82186对照为Ongoing0.801179786、FP602/FN952。旧口径同样是少误报、多漏报，增益区间也跨零。

一月校准集Ongoing为0.901672876；**二月为0.868870502，尚未达到0.9**。两者不能混报。所有成绩都是公开可靠观测构造的代理IoU，不是官方隐藏 `S_queue`。二月被多次用于研究，不是新的独立盲测。

## 训练变化与固定约束

- Onset沿用[原early_full发布包](../best_overall_081796_20261030/README.md)，9个头、历史参考及门槛 `[.50,.50,.50,.55,.55,.55]` 全部不变。
- Ongoing沿用0.82186按道路、三种子的600棵树，每个头继续训练200棵树，学习率0.02，其余63叶/深度7/最小叶60/L2=20不变。保存的24份800树权重已包含原600树，推理无需额外加载0.82186的Ongoing权重。
- 同一8143训练窗口、274维因果特征、节点/已知时距抽样及IPW；不把未知未来当负例。主线截止masked-T、匝道截止T−5，历史参考截止2030-11-30。
- 新训练权重：正例训练窗口的 `q=已知正例单元格数/全部已知单元格数`，将整窗权重乘 `clip(.1/(q+.01),.5,8)`；无已知正例的辅助窗口乘1。再按道路将正例/辅助质量归一回80%/20%。标签只确定**拟合期损失权重**，不参与查询期特征或门控。
- 6—9月拟合，十月从sqrt/inverse两种权重及独立/R127等权log-odds融合中锁定两候选；十一月三种子只报告。R127训练过这些开发日期，因此锚定开发比较并非完全样本外。
- 最终只用6—11月拟合；**一月只选全局门槛0.35**，约束FP/FN各不超过0.82186一月对照的1.05倍、任一道路IoU下降不超过0.03。模型、配置和一月门槛在本轮二月评分前哈希冻结。**没有二月调参。**
- Overall沿用窗口→道路×条件→道路→四类道路等权口径，完整16组时等于 `(Onset+Ongoing)/2`，不以全部单元格汇总IoU替代。

## 原始数据复评与推理

保留本目录、`best_overall_081796_20261030` 及 `r127_reproduction_20261027` 的相对位置。安装本目录 `requirements.txt` 和[R127依赖](../r127_reproduction_20261027/requirements.txt)，获取授权的Kaggle原始数据。

下载并解压[R127完整模型Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)中的 `ConditionalPosition-R127.zip`，SHA256为 `c90229ea2df8215ffe18c52de910b1c32735d569257e36b076b0c3e9e3f07649`。

在本目录运行：

```powershell
python evaluate.py --package "D:/models/ConditionalPosition-R127" --data-root "D:/data/kaggle_public" --work "D:/work/task2-082245" --population both --causality
```

入口核验发布清单、未改动的Onset、R127原文件和原始数组哈希，从原始观测复算两套共777个窗口，逐道路×条件与冻结结果核对到1e-12，并检查改变未来标签、掩码及未来速度不改变预测。验收记录见 `acceptance/`。

已有同顺序274维因果特征时：

```powershell
python code/predict_ongoing_roads.py --models . --features "D:/work/features.npz" --candidate inverse_odds50 --output "D:/work/prediction.npz"
```

输入NPZ含 `X[N,274]`、`panel[N]`、冻结R127概率 `parent[N,6]`；输出概率、门槛0.35与二值预测。顺序见 `feature_names.json`。三种子平均后与R127作等权log-odds融合，所有道路和两套窗口使用同一候选。

`reports/` 保存月份选择、误报漏报、逐道路/时距和区间；`training_archive/` 为实验代码，完整重训还依赖原始数据、初始化模型与中间训练缓存，不声称一键从零重训已验证。此目录不含原始数据或本机环境；文件SHA256见 `manifest.json`。
