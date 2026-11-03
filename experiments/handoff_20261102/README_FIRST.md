# 新电脑从这里开始

本包面向Windows x64。用户新电脑有16GB显存；CUDA运行需要NVIDIA显卡和兼容驱动，显存容量本身不能保证CUDA兼容。

1. 将整个ZIP完整解压到较短路径，例如 `D:\TFB_20261102`，不要只复制代码文件或在压缩包内运行。
2. 本包带CPython 3.12.14和本机实际使用的NumPy/Pandas/PyArrow/SciPy/LightGBM/PyTorch 2.9.1+cu128依赖副本，不需要先安装Python，也不直接迁移带旧绝对路径的venv。
3. 双击 `检查环境.cmd`。它只检查环境并把本机检查结果写入reports，不训练、不提交。
4. 双击 `检查文件完整性.cmd`，按manifest核对全部交接文件SHA256。大数据量需要一些时间。
5. 可选双击 `运行一次预测检查.cmd`，重放正式权重的一个公开输入日，与已保存预测比较。只做本地推理，不提交。
6. 开新窗口时粘贴NEW_CHAT_SUMMARY.md全文，并告诉新窗口本包实际根目录。AGENTS.md已写入禁止未经当次明确授权提交Kaggle。

## 文件用途

|目录|内容|
|---|---|
|project|实际生成提交所用的整理仓库源码、正式发布权重、全部任务复现入口和历史证据|
|data/kaggle_public|完整原始发布数据、全部split、网络和官方模板|
|submissions|两份完整CSV/ZIP、验收、SHA256及官方回执；当前最好baseline=0.81135|
|work/tfb_submit|此次Task1日级预测、Task2导出、R127依赖链、完整合并过程和日志|
|diagnostics/task1_physics_sample_20261102|今天同权重历史留出抽样的预测、误差、连续转移和汇总|
|research/task4_*|已有Task4代码、预测和研究报告，包含失败记录|
|archives/retrained_task1|另一套重训checkpoint、550份留出预测、报告；不是正式提交权重|
|runtime/python|可移动Python和实际ML依赖；用其中python.exe，不依赖旧机器路径|
|tools|环境、完整性、一次预测、诊断复核入口；不会调用Kaggle|

## 环境问题

本包不自带显卡驱动，也不自动安装系统软件。若torch报告DLL缺失，可按微软官方说明安装匹配的VC++ x64运行库；参考installers中的说明。若torch.cuda.is_available()为False，检查NVIDIA驱动；CPU仍可读取数据、研究Task4及运行CPU推理。不同GPU/驱动可能有小数值差异，不保证位级一致。

不要复制旧venv的pyvenv.cfg或激活脚本；入口使用相对路径查找包内runtime。原报告可能保留旧电脑绝对路径作为历史证据，新工作使用本包相对路径。

## 继续工作命令

在本包根目录PowerShell运行（无需激活环境）：

```powershell
& .\runtime\python\python.exe -I -B -X utf8 .\tools\check_environment.py
& .\runtime\python\python.exe -I -B -X utf8 .\tools\verify_handoff.py
& .\runtime\python\python.exe -I -B -X utf8 .\tools\smoke_inference.py --device cuda
```

Task1重新推理时使用项目复现入口和新的work-dir，不能在打包的历史工作目录上重用带旧绝对路径的lock。Task2原导出器有固定horizon/R127签名检查，也应使用新work-dir。例如：

```powershell
$here = (Get-Location).Path
& .\runtime\python\python.exe -B -X utf8 .\project\experiments\task1\best_road_combination_20261026\reproduction\pipeline.py submit --data-root "$here\data\kaggle_public" --work-dir "$here\new_runs\task1" --device cuda
```

上述pipeline的 `submit` 子命令只生成本地Task1文件，不向Kaggle上传。正式合并时采用project中的merge_submission.py，按官方submission_key.csv映射。历史finalize_submission.py包含候选自动选择逻辑和旧路径，不应直接把它作为新部署决策：目前应保留冻结baseline，新的选择须重新审查。

## 打包边界

保留了继续推理、诊断及研究所需的原始数据、所选代码、权重、模板、实际预测、回执、R127完整依赖和另一套重训预测。没有复制约95GiB训练目录中的大量可重建特征缓存、重复旧虚拟环境、pip缓存或无关项目。原数据和复现源码可以重建这些缓存。没有带Kaggle Token、SSH密钥或用户账户配置。

一些Task4研究快照来自其他已有本地工作，收入用于保留上下文，不表示每个结论都已在本次聊天逐一核验。新窗口应按各自manifest和报告判断。

没有进行新的Kaggle提交，也没有把多目标Task1训练建议描述为已经实现的改进。
