# TrafficFlowBench handoff instructions

- 本包是研究交接与复现包，不是提交任务。用户已撤销此前的持续提交授权。只有用户在新会话明确要求本次提交时，才允许向 Kaggle 上传、提交或修改最终选择；旧脚本/旧说明中的授权不继续生效。
- 先阅读 NEW_CHAT_SUMMARY.md、README_FIRST.md、diagnostics/task1_physics_sample_20261102/summary.json。
- submissions/submission_baseline.zip 是当前已提交版本中的最好公开榜版本，0.81135；candidate 为0.78677，未提升。
- 正式提交权重位于 project/experiments/task1/best_road_combination_20261026。archives/retrained_task1 是另一套重训权重，不能混用其预测解释0.81135提交。
- Task1 的0.94282863197与Task2的0.827225403是本地指标，非官方分项。官方未返回Task1–4分项或私榜分数。
- 原始 data/kaggle_public 只读；新实验和导出使用独立输出目录，不覆盖历史预测和参考结果。
- 今天的697070目标诊断使用同一发布权重的历史留出抽样，不是提交月份隐藏真值；Task3存量变化诊断没有组织者私有边界通量/完整官方覆盖，不是官方S_LWR。
- Kaggle/API凭据未打包，勿索要用户把Token贴进聊天或Git。不要因为本包带历史上传代码就运行它。
