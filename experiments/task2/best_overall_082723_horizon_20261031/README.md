# Task2：分时距残差专家，Overall代理IoU 0.827225403

原[0.82639](../best_overall_082639_jaccard_20261031/README.md)及此前版本、最好Onset原样保留。本版在同一冻结900树父模型上训练六个5—30分钟修正头，每头100树；不复制父模型权重，运行需保留0.82639目录、原Onset目录和R127复现目录相对位置。144份残差权重包含新增树，不包含父模型初始logit，必须按本入口组合。

|固定二月窗口|Onset|Ongoing|Overall|Ongoing FP/FN|
|---|---:|---:|---:|---:|
|新367(183+184)|0.776032314|0.878418492|0.827225403|681/417|
|旧410(200+210)|0.738098750|0.806011104|0.772054927|502/1026|

相对0.82639的新口径ongoing增益区间：[-0.00013633609249455819, 0.0036572953912787304]。这是可靠观测代理分数，不是官方S_queue；二月重复开发，非独立盲测，点估计领先不保证稳定提升。

## 固定方法

- Onset9个头和门槛[.5,.5,.5,.55,.55,.55]不变。Ongoing父模型900树不更新，仅训练时距残差。
- 同274维因果输入、8143窗口、全部已知时距和节点IPW，未知未来不当负例。逆密度整窗权重、正例/辅助质量80/20在各时距归一。主线masked-T、匝道T-5，参考截止2030-11-30。
- 残差100树、15叶/深度4/最小叶100/L2=50/lr.03，external init_score=冻结父模型raw logit。保存后预测用sigmoid(parent_raw+alpha*residual_raw)，三种子平均后与R127等权log-odds混合。该组合经独立native测试验证。
- 十月从BCE/Soft-IoU时距头及残差收缩.5/1中锁定两配置；十一月三种子只报告。最终仅6—11月拟合。R127拟合过开发日期，锚定开发非完全样本外。
- 本版bce、alpha=1.0；一月只选最高IoU全局门槛0.375，同时记录FP/FN保护条件可行性。模型、父模型、源代码与门槛哈希冻结后才评本轮二月，没有二月调参。两套窗口同一模型，不按口径切换。

## 原始观测复算

依照0.82639说明安装依赖、获取授权原始数据与R127完整Release，保留相对目录：

```powershell
python evaluate.py --package D:/models/ConditionalPosition-R127 --data-root D:/data/kaggle_public --work D:/work/task2-horizon --population both --causality
```

入口校验模型清单、父链及原始数组哈希，从777个原始窗口重算，逐道路×条件核对到1e-12，未来标签/掩码/速度扰动不改变预测。验收记录见acceptance/。

已有NPZ(X[N,274],panel[N],parent[N,6])时：

```powershell
python code/predict_horizon.py --models . --features D:/work/features.npz --candidate bce_a100 --output D:/work/prediction.npz
```

reports/保存所有对照与逐道路/时距指标，training_archive/记录实验代码；重新训练仍依赖原始数据、父模型和训练缓存，不声称一键从零训练已验证。这里不含原始数据。
