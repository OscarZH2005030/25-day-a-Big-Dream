# Task2：保留救援真阳性的 pending 专家影子候选

本目录发布本聊天截至2026-10-31已通过**重复开发集**保真检查的最好影子候选：冻结partial-label safe veto，加上各日期折一致认定“尚未到达”的pending专家。**不替换R127或仓库的默认模型，不提交Kaggle。**

| 相同12月、1月、合并去重2月窗口 | Overall proxy IoU |
|---|---:|
| R127 | 0.788043545011 |
| 原保留救援链 | 0.790697107913 |
| 本影子候选 | **0.790893192093** |

相对R127绝对提升0.002849647083，即0.284965个百分点。两版2月23个重复窗口只计一次；2015个原始窗口去重为1992个，沿用原评估资格的1479个正未来队列窗口参与Overall。评分为窗口→道路/任务类型→道路→道路族宏平均，含onset与ongoing；负窗口误报单独检查，不把月份分数直接平均。

**不能与仓库[2月单独口径0.817962](../best_overall_081796_20261030/README.md)直接排名。** 它是另一模型、另一窗口范围的分数；本目录没有证明超过该模型。

## 保真检查和限制

- 11月只有onset完整，不混进三个月Overall。相对原保留链，本候选在11月少2个已知FP、12月少1个、新版2月少4个；1月与旧版2月不变。
- 各检查月份没有新增已知真阳性损失，新版2月6/6、旧版2月17/17关键TP保留；未知/低质量单格不能据此断言对错。
- 固定onset阈值0.5、ongoing阈值0.25、二元筛选0.5，无类别增权、道路/月/案例例外。R127阳性与既有unit决策硬保留，unit始终使用未过滤救援作为输入。
- 三个月及11月已反复研究，策略为事后开发组合；只有公开速度代理标签，没有官方隐藏queue truth，不是盲测/官方S_queue。
- 175个另行预冻结的新时刻诊断中，两个影子筛选与原保留链完全相同，未证明新增筛选收益；其中2月没有正onset窗口，不能称跨月验证成功。更大的2585时刻测试在本次发布整理时仍在运行，**未纳入本分数**。
- 142项本地研究测试通过；独立推理适配器在10个旧窗口重建571维特征与预测完全一致，未来目标扰动不影响预测。此包另提供8道路可迁移推理校验。

## 可运行复现

Python3.12；依赖沿用[R127固定环境](../r127_reproduction_20261027/requirements.txt)。原始Kaggle数据和约877MB的R127父链不重复上传；按照[R127说明](../r127_reproduction_20261027/README.md)下载并验证其固定Release。

```powershell
python reproduce.py verify
python reproduce.py metrics
python reproduce.py smoke --data D:/datasets/kaggle_public --r127-package D:/tfb-assets/ConditionalPosition-R127 --work-dir D:/tfb-shadow-work
python reproduce.py predict --data D:/datasets/kaggle_public --r127-package D:/tfb-assets/ConditionalPosition-R127 --work-dir D:/tfb-shadow-work --panel D7_I10_E --origin 2031-02-10T14:50:00Z --output D:/tfb-shadow-work/outputs/example.npz
```

`verify`检查ZIP及每个运行文件SHA256；`metrics`不需要原始数据，直接从固定逐窗口评估记录重算去重1479窗口三组分数；`smoke`在独立解压工作区复核8道路、4个固定策略的逐位预测摘要；`predict`对给定UTC时刻输出[R127、retained、safe_shadow、pending_shadow]的[6,link]布尔数组。选择较短`--work-dir`避免Windows长路径，不覆盖不匹配的既有文件。

`assets/runtime.zip`含本候选所需代码、新增模型和训练统计，约15MB；`source/`为对应可阅读源码；`runtime_manifest.json`固定原始字节；`reports/`保留分数、逐窗口结果、TP锚点及负结果。无需访问原研究目录。标签仅用于评估：推理构造12帧历史及已发布masked-T，传给模型前移除未来标签。按日缓存文件可能包含完整公开训练日，但未来行不进入模型。

本包验收的是**冻结权重推理和评估记录重算**，没有宣称提供全部祖先模型的从零重训；大型训练矩阵未入库。原始竞赛数据、账户凭据与工作缓存未入库。父链/原训练复现入口见R127目录。
