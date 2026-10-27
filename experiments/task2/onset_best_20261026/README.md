# Task2 Onset 最佳参考模型与组会进度

[组会总结](GROUP_MEETING_20261026.md) · [完整模型下载](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task2-onset-best-20261026) · [指标与校验清单](model-catalog.json)

| 模型 | 评价窗口集合 | 综合代理 IoU | Onset | Ongoing |
|---|---|---:|---:|---:|
| Phase367-R41 | 新367窗口 | 0.790703 | 0.721513 | 0.859893 |
| Old410-R21 | 旧410窗口 | 0.754464 | 0.729831 | 0.779097 |

分别保留两个评价窗口集合的最佳参考，不合并模型分数。R41 在旧410窗口的综合代理 IoU 为0.750679，尚未超过 R21。二月已经反复用于开发后复评，不是独立盲测；官方 `S_queue=null`，这里不提供官方队列成绩或排名。

Task1 冻结包历史55天 `S_state=0.942828633`，本次未修改权重、路由和离线预测，也未重测这个历史分数。Task1 完整包沿用[原发布](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/best-road-combination-20261026)，不进入 Task2 特征。

## 下载与预测

Release 附件包含 `Phase367-R41.zip`、`Old410-R21.zip` 和 `SHA256SUMS.txt`。仓库中的 `models/R41`、`models/R21` 保存可审查的代码、配置和完整性记录；运行预测须下载含权重的 ZIP。

解压所选 ZIP，进入对应目录，使用已有 Python 环境及官方公开数据：

```powershell
python -X utf8 verify.py
python -X utf8 predict.py --data-root C:\data\kaggle_public --split validation --output result.csv
```

支持 `--split private`；每个 split 为87000行。依赖版本见包内 `requirements.txt`，Python环境和公开原始数据不随包提供。缓存只写包内 `work`，推理CPU最多4线程；不得修改冻结权重、路由及阈值。

R21 原保存包已经在禁止读取原模型和缓存的条件下精确复现87000行；R41 本次新增相同独立打包验收。验收结果位于各模型 `saved-verification.json`，完整文件 SHA256 位于 `archive-manifest.json`。发布附件校验见本目录 `SHA256SUMS.txt`。

## 核心改进

Onset 采用事件／支持范围与首次起堵时间分解，Ongoing 保持专用状态分支。R41 使用同一首次未来拥堵观测的三相位配对视角训练6类条件时间头，只在固定集中度门控通过时改变时间形状；保留低集中度、已观察排队和 Ongoing 的父策略输出，精确保留每个路段未加未来掩码时的六步最大输出。

训练六月至十一月，十二月选道路路由，一月全局校准。二月仅冻结后重复复评；所有输入只取公开因果历史，未来质量和覆盖率只用于标签／抽样／评分。主要验收与后续拒绝候选见 `reports/`，完整解释见组会总结。
