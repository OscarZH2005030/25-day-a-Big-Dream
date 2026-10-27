# Task2 R127：完整模型、分数复现与新增模型重训

R127在R116上增加I210东向的条件首次发生时间头；保留I10东向、I5北向的R116位置场策略及完整历史父链。Task1与Ongoing权重、路由、输出保持冻结。

位置场是对公开`order_index`的固定坐标投影，不解释为物理邻接、距离或传播速度。推理ZIP保留独立保存时的原始字节和hash；其中发布状态属于保存时记录，当前发布说明以本页为准。

| 固定评价窗口集合 | Task2整体代理IoU | Onset | Ongoing |
|---|---:|---:|---:|
| 二月367个正例窗口 | 0.7959762085498983 | 0.7320598420153661 | 0.8598925750844304 |
| 二月410个正例窗口 | 0.7545993040930380 | 0.7301012853729876 | 0.7790973228130884 |

**这是二月两个不同窗口人口的重复开发复评，不是全年12个月平均，不是独立盲测。官方`S_queue=null`，没有Kaggle官方分数或排名。** 410窗口口径来自768个原始窗口中按固定协议计入的410个正例窗口；两套人口分别计算，不混合。Task1历史55天`S_state=0.942828633`，本次未重测。

本目录提供三项可运行复现：公开validation/private各87,000行预测；从公开观测重新构造缓存并重算上述全部分组指标；重训本轮新增的3个头及3个匹配对照。历史R41及以前父链按实验定义冻结，完整权重和推理代码均在模型包内。

## 1. 下载与环境

[完整Release：模型、精确训练输入及SHA256](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-r127-repro-20261027)

* `ConditionalPosition-R127.zip`：约877MB，包含完整父链代码/权重/冻结路由/配置、预测参考、逐文件manifest及独立验收记录。
* `TrainingInputs-R127.zip`：约1.16GB，包含准确行顺序的训练特征、proxy标签、权重、匹配对照输入及参考模型；只含六月至十一月的训练输入。
* `SHA256SUMS.txt`：两个附件的SHA256；[release-assets.json](release-assets.json)同时固定下载地址、大小与hash。

使用Python3.12，在本目录执行。建议独立虚拟环境；已验收包版本见[environment.json](environment.json)。精确环境保留`torch==2.9.1+cu128`轮子，实际推理和重训使用CPU、最多4线程，验证不需要GPU。

以下PowerShell命令使用用户目录下的短路径，避免Windows长路径限制；macOS/Linux使用相同Python入口并替换为本机目录。

```powershell
$reproAssets = Join-Path $env:USERPROFILE "tfb-r127-assets"
$reproWork = Join-Path $env:USERPROFILE "tfb-r127-work"
$reproData = "D:/datasets/kaggle_public" # 替换为本机公开数据目录
python -m venv .venv
$reproPython = Join-Path (Get-Location) ".venv/Scripts/python.exe"
& $reproPython -m pip install -r requirements.txt
& $reproPython fetch_assets.py --output "$reproAssets"
& $reproPython "$reproAssets/ConditionalPosition-R127/verify.py"
```

下载器同时检查SHA256、ZIP CRC及解压路径，检查Windows路径长度，并记录完整解压标志。已有附件可用`--local-assets <附件目录>`离线校验解压；`--output`可指定缓存目录，默认用户目录下`tfb-r127-assets/`。工作缓存写到指定复现目录或包内`work/`，不改冻结权重。

公开原始数据请从[IEEE Traffic Flow Bench Kaggle Data](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data)获取并按竞赛要求授权下载；也可在本机配置Kaggle CLI后执行`kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench`。`$reproData`必须指向同时包含`corridors/`和`task2/`的目录。上述命令直接调用虚拟环境Python，无需运行PowerShell激活脚本或更改执行策略。账号凭据和原始竞赛数据不放入仓库。

## 2. 复现公开模板预测

```powershell
& $reproPython "$reproAssets/ConditionalPosition-R127/predict.py" --data-root "$reproData" --split validation --output "$reproAssets/ConditionalPosition-R127/work/validation.csv" --io-audit --audit-output "$reproAssets/ConditionalPosition-R127/work/validation-audit.json"
& $reproPython "$reproAssets/ConditionalPosition-R127/predict.py" --data-root "$reproData" --split private --output "$reproAssets/ConditionalPosition-R127/work/private.csv" --io-audit --audit-output "$reproAssets/ConditionalPosition-R127/work/private-audit.json"
```

每份应有87,000行；与包内`predictions/queue-validation.csv`、`queue-private.csv`按列、键顺序及`queue_pred`逐行一致。CSV换行格式可以随系统不同，比较数据行。启用`--io-audit`时输出请放在包内`work/`，审计会禁止访问原实验模型/缓存，并要求所有原生模型文件来自解压包。

这些是公开模板的预测复现，March/April隐藏queue标签没有公开；本入口不提交Kaggle。

## 3. 重算两套代理分数

```powershell
& $reproPython evaluate.py --package "$reproAssets/ConditionalPosition-R127" --data-root "$reproData" --work "$reproWork/evaluation" --causality
```

脚本用公开训练观测与公开masked-T重建日缓存和网络，不读取作者机器上的历史缓存。重建数组逐项检查dtype、shape、内容SHA256，再读取[indices](indices/)固定窗口。按包内原评分器聚合，逐个panel/condition与[expected-metrics.json](expected-metrics.json)核对，误差需小于`1e-12`。

`--causality`检查两套人口各16组：主线未来扰动、匝道T及未来扰动、future y/m/has_future_queue扰动均不得改变预测。未来质量和coverage只用于标签、训练抽样与评分。推理不使用未来/双向离线Task1输出。

模型阈值固定Onset `.5`、Ongoing `.25`。训练用2030年6—11月；早期试验只用6—9月拟合、10—11月留出；12月选择可用道路，1月仅全局校准，2月冻结后重复开发复评。原始观测代理标签使用公开速度阈值、eligibility及quality规则，不是官方`queue_true`。

## 4. 精确重训新增模型及匹配对照

```powershell
& $reproPython fetch_assets.py --output "$reproAssets" --training
& $reproPython retrain.py --inputs "$reproAssets/TrainingInputs-R127" --output "$reproWork/retrained" --models all
```

默认串行重训6个模型，每100轮保存断点，原权重不会覆盖：

| 模型 | 输入 | 原生轮数 | 目标 |
|---|---:|---:|---|
| R116 I10东向位置场及随机场对照 | 914 | 各600 | 六个未来步的cell BCE |
| R116 I5北向位置场及随机场对照 | 914 | 各600 | 六个未来步的cell BCE |
| R127共享条件首次时刻头及随机场对照 | 916 | 各600、六分类 | 已确定未来拥堵事件的首次步 |

`--models selected`只重训3个采用模型；`--models controls`只重训3个对照。所有原始X/y/w的行顺序、dtype、数据hash、完整参数、seed以及期望模型hash见[training/input-manifest.json](training/input-manifest.json)。输入来自当时保存的精确训练数组，无需作者本地路径；不把缺失未来标签当负例。每个输出都必须与冻结模型SHA256逐字节一致，失败时脚本会明确报错。该严格训练校验以记录的原始环境为准；跨编译器/系统请先验证冻结包预测和评分，不能把不匹配权重宣称为原权重。

冻结父链已经完整包含在推理包内；本入口复现本轮增量训练，不修改或重新选择历史父模型。训练原脚本、固定数据切分、IPW抽样行表、标签分解与校准协议分别放在[training/original_sources](training/original_sources/)和[indices](indices/)；保留原脚本用于逐行审查，历史脚本含原机器guard/截止日期，实际可迁移重训入口为本目录`retrain.py`。

也可以跳过约1.16GB的训练输入附件，从公开观测重建全部15份训练数组：

```powershell
& $reproPython rebuild_inputs.py --package "$reproAssets/ConditionalPosition-R127" --data-root "$reproData" --output "$reproWork/rebuilt-inputs" --work "$reproWork/rebuild-cache"
& $reproPython retrain.py --inputs "$reproWork/rebuilt-inputs" --output "$reproWork/retrained-from-raw" --models all
```

该入口把冻结的抽样行索引和IPW/窗口权重作为原实验协议，直接从公开观测生成因果特征和proxy标签；对重建的15份NPY逐字节检查是否等于训练附件。当前/未来质量只参与相应训练标签与抽样，模型特征严格使用公开因果输入。原始日缓存重新生成并逐项检查内容hash，禁止读取作者旧缓存；输出目录必须是新的短路径，不会覆盖原模型。

## 5. 验收与当前限制

[reports](reports/)保存December开发、一月全局校准、完整性/因果测试、独立保存和本次解压复现报告。发布前已在隔离目录通过：6个模型重训hash一致；两份公开模板174,000行一致；两套评分全部分组一致；32组因果扰动通过；原模型/缓存IO被阻断；从原始公开观测重建的15份训练数组与附件逐字节一致。

公开发布只有2030年6月—2031年2月九个月未遮蔽观测；3—4月只提供遮蔽输入，5月未发布。因此没有12个月平均分。[月份审计](reports/available-months.json)。

Onset当前同时有事件漏检、时间边界错误和旧置信门控覆盖限制。最新三种后续训练方案未通过预设提分门槛，未替换R127。误差与oracle记录是未来标签参与的诊断，不能当作可部署模型效果；不根据二月逐路调参或回滚。

本次发布只更新Task2可复现工件，Task1原92文件保持冻结。本目录不发布官方排名、不代提交竞赛、不自动恢复夜间训练。

## 6. Onset仍低的具体原因与下一步

当前R127的December诊断使用177个Onset正例窗口，与上面的二月367/410人口分开：518个漏报的道路×时间格中，267个（51.5%）来自该link整段未来预测峰值低于`.5`，251个（48.5%）来自已经预测到事件但时间不对。334个误报格中216个（64.7%）位于确实发生拥堵的link，主要问题也是报早/报晚。以上是相关窗口中的cell计数，不是独立事件比例或因果归因。

另外，在完整标签、历史未观测拥堵、且已被预测支持的189个link-view子集中，101个因旧R40条件概率最大值低于`.5`而禁止新时间头替换父曲线。这说明新头分类变好未必能传递到最终输出；这个子集不能外推到所有拥堵，也不能据此直接取消门控。未来标签oracle仅用于诊断，数值不是可实现的模型成绩。

后续研发建议按不同错误分别处理，并保留独立对照：

1. **事件漏检为主的道路优先做检出。** December中I10东向86/127个FN、I405北向35/44个FN来自事件未检出。应测试公开因果位置场支持的事件发生头及明确负例，检查提高召回是否引入新误报；继续只改首次时刻无法恢复这些漏报。
2. **错时为主的道路优先做边界与决策链。** I210西向、I405南向、I5南向的FN更多来自时间错误。可研究独立因果“当前状态是否确定”的信息，检验能否安全替代旧模型置信门控；Task1原包继续冻结，禁止引入未来/双向离线Task1预测，不直接下调旧门槛。首次时刻仍不确定的正例可研究区间标签：未知前缀不强行当负例，未来缺失信息只进入训练标签/评分。
3. **逐路分支本身已经做过对照。** R131道路残差、R133精确prefix正例扩充、R138条件累计CDF三种方案均未过原提分门槛。后续应改变可识别信息或目标分解，避免重复扫这些方案的参数。
4. **逐月方案应单独立协议。** 可对现有九个月分别作月内时间留出，按月报告并给等月权平均；这是新的评价集合，不能与现有二月窗口分数横比。缺失的3—5月完整观测不能用遮蔽输入补成十二个月。

上述为待验证方向。本次发布没有自动启动这些新实验，也没有改动当前最佳模型。
