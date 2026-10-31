# Task2：固定最好Onset，Overall代理IoU 0.821855853

本版使用原样保留的 `early_full` Onset，与新的按道路训练 Ongoing 模型组合。新增24个LightGBM权重（三种子×八方向）、Ongoing历史参考、因果特征代码、固定配置和评估记录均在此目录。Onset权重沿用[0.817962444版本](../best_overall_081796_20261030/README.md)，原版不修改。R127完整父链继续沿用原Release。

| 相同二月窗口 | Onset | Ongoing | Overall | Ongoing FP / FN |
|---|---:|---:|---:|---:|
| 新367（183+184） | 0.776032314 | **0.867679393** | **0.821855853** | 789 / 377 |
| 旧410（200+210） | 0.738098750 | **0.801179786** | **0.769639268** | 602 / 952 |

两套人口使用同一组权重与门槛，不按道路、窗口口径切换模型。Overall仍按“窗口→道路×条件→道路→四类道路”等权汇总；完整16组时等于 `(Onset + Ongoing)/2`。全单元格汇总IoU与此指标不同。

新口径Ongoing对照：原R127为0.859892575、FP922/FN372；上一最好共享模型为0.861281251、FP914/FN380。新模型相对上一最好减少125个FP、3个FN；相对R127减少133个FP、增加5个FN。旧口径原R127为0.779097323、FP765/FN960。

这些是可靠公开观测构造的**代理IoU**，不是官方隐藏 `S_queue`。二月曾多次用于研究，不能视为新的独立盲测。相对上一最好，新口径增益0.006398的按日分块95%区间为[-0.009341,0.019833]，仍跨零；旧口径增益0.020299的区间为[0.014118,0.026644]。保留点估计领先版本不代表已经确认稳定提升或达到0.9。

## 模型和月份约束

- Onset：不改9个既有头、历史参考与门槛 `[.50,.50,.50,.55,.55,.55]`。
- Ongoing：每条道路独立训练共享5—30分钟时距的二分类头，274维输入；600轮、63叶、深度7、最小叶60、L2=20、学习率0.04。三种子20261030、20261031、20261101平均概率，然后与未改动R127作等权log-odds融合。
- 输入为过去可靠速度、占有率、容量归一化流量、队列持续/进出、真实上下游边、匝道、位置/时段参考及整条道路场描述。主线截止masked-T，匝道截止T−5。历史参考固定2030-11-30。
- 同一8143个6—11月Ongoing训练窗口，包含4039个有已知未来正例窗口和4104个无已知未来正例窗口；沿用节点/时距抽样与IPW。未来缺失只作掩码，不当负例。
- 6—9月开发拟合，十月比较两种容量和独立/融合配置，锁定候选；十一月三种子报告，不重选配置。R127训练过十月和十一月，锚定开发比较不是完全样本外。
- 最终6—11月拟合，**只用一月选择全局门槛0.325**。模型、配置及门槛在本次二月评分前哈希冻结；没有二月调参。

## 从公开原始观测复评

保留仓库中本目录、既有Onset目录和 `r127_reproduction_20261027` 的相对位置。安装本目录 `requirements.txt` 与[R127环境要求](../r127_reproduction_20261027/requirements.txt)。原始Kaggle数据需自行授权获取，仓库不含原始数据或本机Python环境。

下载并解压[R127完整权重](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)中的 `ConditionalPosition-R127.zip`；SHA256为 `c90229ea2df8215ffe18c52de910b1c32735d569257e36b076b0c3e9e3f07649`。

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python evaluate.py --package "D:/models/ConditionalPosition-R127" --data-root "D:/data/kaggle_public" --work "D:/work/task2-082186" --population both --causality
```

评估入口直接从原始观测重建因果输入，核验本目录、原Onset及R127原文件清单、原始数组哈希，在同777个正例窗口上重新预测，逐道路×条件与保存分数比较到1e-12；同时检查改变未来标签、掩码和未来速度不改变预测。输出 `metrics.json`、`verification.json`。验证记录见 `acceptance/`。

所有新头在R127安装原模型加载保护前加载，Ongoing原父链不修改。`code/hybrid_ongoing.py` 是两条件原始观测推理适配器。历史参考在推理时不更新，不读取一月或二月标签。

已有相同274维因果特征时，也可独立调用：

```powershell
python code/predict_ongoing_roads.py --models . --features "D:/work/ongoing_features.npz" --candidate larger_odds50 --output "D:/work/ongoing_prediction.npz"
```

输入NPZ包含 `X[N,274]`、道路名称 `panel[N]`、冻结R127概率 `parent[N,6]`。特征顺序见 `feature_names.json`；输出概率、固定门槛0.325和二值预测。八道路共256行的独立特征入口与实验概率逐位一致，见 `acceptance/feature_runtime.json`。

`reports/` 保存月份选择、逐道路/时距、误报漏报和不确定性结果。`training_archive/` 保存实验代码及原路径布局；完整重新训练仍依赖原始数据和训练缓存，不声称已经验证一键从零训练。`manifest.json` 记录本目录全部交付文件SHA256。
