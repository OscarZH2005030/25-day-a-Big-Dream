# Task 1：0.942828633组合的完整复现流程

本目录把**原始公开数据→从零训练→评估→Task 1预测→完整提交合并**连起来。历史成绩来自共同55天、6,636,705个官方目标；这是已经查看过并用于逐路选模的本地验证，**不是独立盲测、Kaggle官方成绩或全赛总分**。

完整包（含权重）：[Task1复现Release](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task1-repro-09428-20261027)。Git保存源码、配置、中文说明、数据指纹与验收；权重保存在Release附件，原始比赛数据需从Kaggle自行获取。

## 1. 环境与数据

- Windows 10/11 x64、Python 3.12；从零训练Jayden分支需要CUDA GPU，已验证环境为RTX5080 Laptop 16GB。推理/评分可以选CPU。
- `environment-reference.json`是本机实际版本。新环境可运行同目录`准备环境.ps1 -PythonPath 'C:\Python312\python.exe'`，仅创建本项目`.venv-task1-repro`，不修改共享或系统Python，不安装系统CUDA Toolkit。需要网络下载依赖。
- 官方数据根必须是含`config/corridors/task1`的`kaggle_public`本身。原数据只读；缓存、检查点、预测均写入独立`WorkDirectory`。
- 中文PowerShell入口采用UTF-8 BOM，兼容Windows PowerShell 5.1；使用`powershell.exe`，无需`pwsh`。

```powershell
# 在完整包根目录执行；请替换数据和基础Python路径。
powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduction\准备环境.ps1 -PythonPath 'C:\Python312\python.exe'
$python = (Resolve-Path '.\.venv-task1-repro\Scripts\python.exe').Path
# 已有匹配环境时可跳过安装，直接把$python设为那个环境的python.exe。
$data = 'D:\datasets\kaggle_public'
$work = Join-Path $PWD 'runs\fromscratch'
```

## 2. 冻结路由和三种训练配方

| 道路 | 最终采用 |
|---|---|
| D12_I405_N、D12_I5_N、D12_I5_S、D7_I10_E、D7_I10_W、D7_I210_W、D7_I405_N、D7_I405_S | Jayden T1＋186数值特征/186 finite标记 |
| D12_I405_S | 十道路全局186特征LightGBM→本路适配 |
| D7_I210_E | 原102输入图Transformer；原图无非自身主线邻居的节点，两通道回退51特征LightGBM |

这不是512宽12层大模型。所选Jayden与原图Transformer均为64宽、3层、4头；Jayden为225,958参数，原图Transformer为208,678参数。

**Jayden八路：**历史28日2030-06-01至06-28；监督168日06-29至12-13；内部选模22日12-14至2031-01-04；最后55日01-05至02-28。历史画像只取前28日的公开原始训练观测。24步窗、64节点patch、batch2；每路seed2026、最多60轮×1024更新、patience8；AdamW 3e-4→3e-5余弦、weight decay0.01、clip1、float32。186数值先signed-log/5、float16落盘，再用训练原视图的均值方差归一化并截断±8，拼接finite。每天固定一份合法遮挡视图，随机窗口50%采用该视图；隐藏10%已发布合格观测，最多替换20%监督目标且总数不变，遮挡后重算全部特征和基底。不是原Jayden逐步重新遮挡A_dynamic的复现。

两路与后续六路分别使用`vendor/jayden2`和`vendor/jayden8`，保留原ROADS索引及后六路的`+2`种子偏移；仅跳过最后没采用的D12_I405_S、D7_I210_E两个Jayden候选。每路独立初始化，因此跳过未选路不改变其他路配方。

**186树模型：**204个训练日06-01至12-21，69个选模日12-22至02-28；历史画像仅取204日的masked已发布观测。372特征＋10道路one-hot＋1相对位置，共383列；训练/选择每道路日最多1500/1000目标，逐日等权。全局两通道从零最多600轮、学习率0.04；D12_I405_S从同一全局模型最多再加180轮、学习率0.015，早停45；63叶、max_bin127、seed20261024。若适配在选模子集更差，保留零新增树的全局模型。仅训练最终所需的树分支，不训练已淘汰的小Transformer。

**D7_I210_E原模型：**同28/168/22/55划分，102输入，64/3/4，24步、64节点、batch2，60×1024、patience8，seed2026；无新增物理损失，FD/conservation权重均为0。原51特征树使用全部168日官方目标、22日RMSE早停，最多2000轮、patience100、31叶、学习率0.05。标签保持原流程“按25/600归一化float32后还原”的精度，再拟合物理单位残差。`vendor/original51`保留上游worker及作者注释，历史证据保存原命令和参数；旧路径和resume文件在新运行中重新定位，绝不依赖旧机器检查点。

不同配方的训练/选模日期和预算不同，不能把此对比解释为严格同预算消融。55日曾用于路由选择，路由在本次复现中冻结，不用逐点真实误差路由。

## 3. 从头训练

```powershell
# 全链路小检查：真实数据、前反向/树拟合验收；不产出正式训练分数。
& $python -X utf8 -u .\reproduction\pipeline.py train-all --data-root $data --work-dir (Join-Path $PWD 'runs\smoke') --device cuda --smoke

# 全部必要模型从零训练，完成后组装到 $work\trained-bundle。
& $python -X utf8 -u .\reproduction\pipeline.py train-all --data-root $data --work-dir $work --device cuda
```

也可以使用中文入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduction\运行复现.ps1 -Stage Train -DataRoot $data -PythonPath $python -WorkDirectory $work
```

阶段顺序：`audit → history → jayden2 → jayden8 → global186 → original51 → assemble`。每阶段可用同名`pipeline.py`命令独立执行。数据审计核对历史保留的5,540个训练/网络文件SHA256；模板键另行逐路核验。

子进程完整输出保存于工作目录`logs`，失败命令及日志路径记录在`logs/last-failure.json`。从零路径不会读取Release中的旧权重；Git克隆后仅复核已发布权重时，先运行：`powershell -NoProfile -ExecutionPolicy Bypass -File .\reproduction\下载发布权重.ps1 -PythonPath $python`。

Jayden每5轮/早停/结束保存last.pt，支持原始优化器和RNG状态续训；原Transformer使用上游last.pt恢复。树模型复用完整已保存模型；训练中尚未完成的一棵树模型需要重新拟合。小检查树写入独立`trees-smoke`，不会误当正式模型。工作目录文件锁防止同目录重复计算。更换数据/源码/配置请新建目录，不把旧完成标记用于新实验。

## 4. 重新评估并记录实际分数

```powershell
# 对刚训练的权重评估，不会借用发布权重或复制历史分数。
& $python -X utf8 -u .\reproduction\pipeline.py evaluate --data-root $data --work-dir $work --bundle (Join-Path $work 'trained-bundle') --device cuda

# 若只复核已发布的0.9428权重，省略 --bundle，另用独立工作目录：
& $python -X utf8 -u .\reproduction\pipeline.py evaluate --data-root $data --work-dir (Join-Path $PWD 'runs\published') --device cuda
```

报告在`evaluation/train/metrics.json`。每种R1/R2/R3内按全部目标SSE计算速度RMSE及**每车道**流量RMSE，再算`0.54*max(0,1-speed_RMSE/25)+0.46*max(0,1-flow_RMSE/600)`；三regime等权，再十方向等权（等价五家族等权）。完整目标数必须6,636,705。

**验收边界：**本次交付重新评估发布权重，并执行冷启动组件/小检查；不把它称为又从零完成一遍全部60轮正式训练。新机器、新版本和CUDA算子可能产生数值差异，从头重训必须以新报告为准，不能保证精确回到同一小数位。`historical-evidence`是旧实验证据，`acceptance`是本次实际验收，两者分开保存。

## 5. 生成Task 1预测部分

```powershell
# 从零训练的模型
& $python -X utf8 -u .\reproduction\pipeline.py submit --data-root $data --work-dir $work --bundle (Join-Path $work 'trained-bundle') --device cuda

# 使用已发布权重：省略 --bundle，也可 --device cpu。
& $python -X utf8 -u .\reproduction\pipeline.py submit --data-root $data --work-dir (Join-Path $PWD 'runs\published') --device cuda
```

输出`submissions/validation/<道路>.csv`、`private/<道路>.csv`及合并的`state_submission.csv`，合计**6,735,795行**。推理时拒绝读取未掩码真值，只按官方Task1模板取目标；逐键、有限非负、全部日期和行数验证。可复用已生成的日期分片，但输入签名变化时拒绝复用。

## 6. 生成真正可上传的完整Kaggle文件

Task 1文件单独不是全赛提交。Task 3直接评估Task 1的状态，不需要另填。准备真实Task2二值队列和Task4非负路径流量预测后：

```powershell
& $python -X utf8 .\reproduction\merge_submission.py --data-root $data --state-dir (Join-Path $work 'submissions') --queue 'D:\results\queue-validation.csv' --queue 'D:\results\queue-private.csv' --odme 'D:\results\odme-all.csv' --output (Join-Path $work 'submission.csv')
```

queue输入须含`panel,window_id,timestamp,link_id,queue_pred`；odme须含`panel,departure_time,path_id,path_flow`，departure_time按原period token保留。可重复`--queue/--odme`传多个文件。

合并器按`submission_key.csv`的submission_id顺序输出`submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow`。缺预测、重复键、非有限值、负值、非二值队列或多余输入均报错。当前公开数据应有state6,735,795、queue174,000、odme70,708，合计6,980,503行。不会用Task2/4全零占位伪造已完成的全赛提交，也不会自动上传Kaggle。

## 7. 文件和来源

- 根目录`selected-road-models.json`：发布权重的冻结路由和哈希；`training-origin.json`仅出现在新训练组装包，实际分数待评估。
- `vendor`：实际训练源码快照；路径移植和跳过未选路的改动记录在新运行`source-portability.json`。
- `historical-evidence`：日期、训练参数、原模型命令、逐路历史成绩；不是本次重训记录。
- `acceptance`：本次组件、全55日评估、全部提交部分的验收。
- 对源码贡献保留原作者；本次Git提交和合并使用用户Stevexxx / chengweiliu669-pixel身份。
