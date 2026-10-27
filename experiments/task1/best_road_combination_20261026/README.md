# Task 1 最佳逐路组合与完整复现

历史共同55日S_state **0.942828633**。8路Jayden186、1路适配LightGBM、1路原Transformer及孤立点回退；这是已查看过的选模验证，不是Kaggle官方总分。

**[中文完整复现流程](reproduction/README_完整复现.md)**：从原始公开数据重新构建历史/特征、训练三种配方、冻结逐路路由、评估、生成全部Task1预测，再接入Task2/4生成可上传的完整submission.csv。

- 从零：`python -X utf8 reproduction/pipeline.py train-all --data-root DATA --work-dir RUN --device cuda`
- 新模型评估：`python -X utf8 reproduction/pipeline.py evaluate --data-root DATA --work-dir RUN --bundle RUN/trained-bundle --device cuda`
- 新模型提交部分：将上面的`evaluate`改为`submit`。
- 复核发布权重：省略`--bundle`；Git克隆后先运行`reproduction/下载发布权重.ps1`，从零训练不需要旧权重。

[完整代码和权重复现包](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/task1-repro-09428-20261027)。原始比赛数据不包含在Git或权重包内。

`reproduction/historical-evidence`保存历史来源，`reproduction/acceptance`保存本次真实验收。完整从零60轮训练未在这次流程补齐任务中重复执行；新机器训练的实际分数以其新评估报告为准。
