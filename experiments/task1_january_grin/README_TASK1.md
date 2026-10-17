# Task 1：状态重建实验

新增的多跳邻居／拥堵变化独立消融及特征缓存，见 `README_TASK1_EXPERIMENTS.md`。
原命令保留；新实验请使用 `run_task1_experiments.ps1`。
三个特征组的残差／直接预测对照及自动实验日志，见 `README_TASK1_TARGETS.md`，
入口为 `run_task1_targets.ps1`。

项目目录：`C:\Users\DELL\Desktop\UW\IEEE比赛`。

数据根目录：`2026-ieee-big-data-traffic-flow-bench\kaggle_public`。
命令从项目目录运行；代码不改写原始数据。所有输出在项目内，已有结果不会覆盖。

## 先运行什么

在 PowerShell 中：

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'

# 依赖和测试（本机已创建环境；换电脑时执行）
powershell -ExecutionPolicy Bypass -File .\setup_task1.ps1

# 抽查一个方向：检查所有日期分区，每个 split 优先抽查 R1/R2/R3 各一天
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode audit

# 14 天训练、3 天开发集、50 轮：仅检查全流程，不是正式实验结论
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode smoke

# 第一次正式开发实验：一个方向，6–12 月训练，完整一月开发集
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode dev -Threads 4
```

脚本会打印结果目录。先查看该目录下的 `metrics_summary.csv` 和
`metrics_by_panel_regime.csv`。不要根据 smoke 的分数选择模型。

十个方向运行时，把 `-Panels all` 加到 audit/dev 命令末尾。按方向顺序训练，
单方向结果完成后即保存，不会将十个方向训练样本同时放进内存。
长实验可在终端按 Ctrl+C 停止；当前版本不自动恢复中断的方向。
需要补跑时为剩余方向指定新输出目录，不覆盖已有结果。

## 方法与验证边界

- B0：按 link × 星期 × 五分钟时段求历史均值；空时段退回 link 均值，
  再退回训练期全局均值。仅 >=75% observed 且速度/流量有限的记录进入统计。
- B1：同站点前后可见值的线性插值，默认两侧观测总跨度不超过 60 分钟；
  短距离单侧观测（15 分钟内）兜底，再退回 B0。跨日先采用历史兜底。
- B2：在 B1 初始预测上学习残差；两个原生 LightGBM 回归器分别学习速度和
  每车道流量。先用固定 350 轮；不对二月测试集 early stopping。
- 特征包括前后可见值、时间间隔、±5/15/30 分钟观测、历史偏差、
  上下游最近可观测主线路段及其 ±15 分钟观测、路段参数、时段、质量。
- 邻居根据主线拓扑搜索，跳过无观测中间路段、排除匝道/connector；不按 ID 排序。
- 当前版本包含基础时空特征；可调传播时延、匝道流量、物理约束及融合是后续消融。
- 每天均匀抽取最多 1000 个模板目标用于残差训练；开发集评分使用全部目标。
  这是计算预算设置，未对 R1/R2/R3 做目标函数重加权，后续可作为消融。
- 模型输入仅从 masked 数据构建。完整 train 只用于训练标签、历史统计和评测。
  训练样本的历史统计排除当日全部记录；一月、二月不进入开发模型历史统计。
- Task 1 是离线重建，允许使用评估月份的前后可见观测。这套特征不可直接复用
  到 Task 2 在线预测；Task 2 必须另设时间边界。
- 2030/06–12 拟合，2031/01 开发，2031/02 最终留出测试。日期是合成日历。
- 模板是唯一目标集合，额外空白不评分；重复、缺行、NaN、负预测会报错。
  速度 RMSE 和每车道流量 RMSE 先按 regime 计算，再等权平均 regime、方向、走廊。
- 当前只做非负输出约束；没有加入 Task 3 物理优化，也不声称本地物理分数有效。

## 二月留出测试

先在一月选定方案，再执行一次；替换 `dev_实际目录`：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode test -Models '.\runs\dev_实际目录\models'
```

多方向模型对应加 `-Panels all`。测试命令复用模型，不重新拟合。

## 最终重新拟合与生成 Task 1 文件

方案冻结后执行；不要提前把二月并入开发训练。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode final-fit -Panels all
powershell -ExecutionPolicy Bypass -File .\run_task1.ps1 -Mode predict -Panels all -Models '.\runs\final-fit_实际目录\models' -Method B2_lightgbm
```

predict 默认同时覆盖 validation 和 private，输出带自然键的 state CSV。
若一月/二月选择 B1，改为 `-Method B1_interpolation`。
该文件还需要与 Task 2、Task 4 合并，**不能直接作为完整 Kaggle 提交**。
最终合并应以发布的 `submission_key.csv` 和 `sample_submission.csv` 校验。

## 直接 Python 参数

```powershell
.\.venv\Scripts\python.exe -m task1 --help
.\.venv\Scripts\python.exe -m task1 run --help
.\.venv\Scripts\python.exe -m pytest tests -q
```

可通过 `--sample-per-day`、`--rounds`、`--max-gap-minutes`、`--threads` 设置预算；
每次变更用新 `--output`。每个 run 保存配置、依赖版本、源文件哈希、模型、
特征重要性、逐日误差、拥堵分组误差及有限数量的示例记录。
`all_regimes_present=False` 代表样本日期没有覆盖全部 regime，该分数只是子集诊断。
单方向汇总也只是该方向结果，不能宣称为比赛总体成绩。
当前版本不自动画图，不做统计置信区间；相关误差表已导出供后续分析。

本地 `profile.pkl` 仅加载本项目自行生成的文件，不加载来源不明的 pickle。

## 参考

- 发布数据中的 `README.md`、`config/competition_contract.json`。
- 官方仓库：https://github.com/jacky850/trafficflowbench-public
- 官方评测：https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview/evaluation
