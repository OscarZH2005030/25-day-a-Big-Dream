# Task 1：两组独立特征实验与缓存

从项目目录运行。环境和依赖沿用第一版，无需新增安装。

## 第一次正式运行

```powershell
Set-Location -LiteralPath 'C:\Users\DELL\Desktop\UW\IEEE比赛'
powershell -ExecutionPolicy Bypass -File .\run_task1_experiments.ps1 -Mode dev -Threads 4 -ReferenceRun '.\runs\dev_20261016_160422'
```

一次顺序完成三个版本：

| 方法 | 特征 | 对照关系 |
|---|---|---|
| B2_lightgbm | 原始 50 个特征 | 重新拟合当前 B2，校验能否复现原结果 |
| B2_multihop | 原始特征＋36 个多跳空间特征 | 只改变空间特征，不加入拥堵变化组 |
| B2_congestion | 原始特征＋25 个拥堵变化特征 | 只改变变化／归一化特征，不加入多跳组 |

三者固定相同的 214 天训练、完整一月评测、每天 1000 个训练样本、
随机种子、350 轮和四线程。B0/B1 也一并导出作为参照。
两组新增特征尚未证明有效；最终看正式开发实验结果。

`-ReferenceRun` 指向第一次正式 B2 实验。程序核对训练/评测区间、
抽样数、轮数、线程及随机种子，再逐 regime 核对原 B2 与重算 B2 的
目标数量、两项 RMSE 和 S_state（数值容差 1e-7）。不一致时写出检查结果并报错，
避免把流程变化误认为特征收益。该参数可省略；换其他 panel 时，需选择包含对应
panel、相同配置的参考实验，或省略该参数。

## 新增特征的定义

- 多跳按拓扑中遇到的可观测主线路段计数，加入第 2、第 3 个邻居。
  无观测中间路段增加距离但不消耗观测跳数；排除 connector/匝道，保护循环。
  多分支取按路段中心路径距离倒数加权的可见观测，缺失权重不参与归一化。
  上下游分别加入同刻和 ±15 分钟速度／每车道流量，以及覆盖率和距离。
- 拥堵变化组只从原 B2 的 masked 特征计算：速度/自由流速度、流量/每车道容量、
  前后观测变化率、过去/未来斜率、上下游差值和邻居时间变化。
  queue margin 是可见速度代理的阈值距离，不是官方隐藏队列标签。
- Task 1 允许离线前后观测；这套特征不能直接用于 Task 2 在线预测。
- 训练仍然排除当天完整数据计算历史统计。评测特征缓存不读取完整评测标签。
- 二月测试集未在此命令中使用。

## 特征缓存

默认缓存位于项目内 `cache\task1\<内容哈希>`。
第一次显示 `[cache] MISS`，生成训练和评测特征；后续相同数据及特征代码显示
`[cache] HIT`。三个模型从相同的缓存样本选择不同特征列，避免重复生成。

缓存身份包含：方向、确切日期列表、抽样数和随机种子、插值跨度、历史统计策略、
特征实现源代码及数据处理依赖版本；原始 masked/unmasked 训练文件、模板、
网络文件均做完整内容哈希。评测缓存还绑定训练缓存和历史 profile。
改变上述条件会生成新的缓存目录。轮数/线程和选用哪个特征组不影响特征缓存身份。
缓存命中仍要读取原始文件计算哈希并核对缓存文件完整性，不是零 IO。

缓存先写入独立临时目录，完成校验信息后原子发布；未完成的临时目录不会命中。
缓存文件缺失、损坏或被修改时程序报错，不静默使用旧文件。
需要避开损坏缓存时指定新的项目内目录，例如 `-CacheDir 'cache\task1_rebuild'`。
本地 profile.pkl 仅用于本项目生成的可信缓存。

新实验模式使用缓存；原 `run_task1.ps1 -Mode dev` 仍是原流程，不调用新缓存。

## 结果查看

每次生成新的 `runs\ablation_dev_时间戳`，不覆盖之前的实验。

- `comparison_vs_B2.csv`：三个版本及 B0/B1 的分数、相对 B2 的增益、宏平均 RMSE。
- `metrics_by_panel_regime.csv`：R1/R2/R3 分项指标。
- `evaluation\D12_I5_N\*_congestion_errors.csv`：拥堵/非拥堵分组误差。
- `evaluation\D12_I5_N\*_gap_errors.csv`：按前后可见观测总跨度分组。
- `evaluation\D12_I5_N\*_daily_errors.csv`：逐日配对比较。
- `cache_usage.json`：缓存键和是否命中；`reference_check.json`：旧 B2 复现检查。
- `models\baseline|multihop|congestion\D12_I5_N`：各自模型及特征重要性。
- `COMPLETE.json`：整个批次正常完成的标记；没有该文件代表未完整完成。

筛选有意义的改进时，同时看总体、R3、拥堵和长缺口，不只看单个最高分。

## 单独跑一组、或先做小规模检查

```powershell
# 单独比较 B2 与多跳组，baseline 会自动加入
powershell -ExecutionPolicy Bypass -File .\run_task1_experiments.ps1 -Mode dev -Variants multihop -Threads 4 -ReferenceRun '.\runs\dev_20261016_160422'

# 单独比较 B2 与拥堵变化组；相同配置会复用同一特征缓存
powershell -ExecutionPolicy Bypass -File .\run_task1_experiments.ps1 -Mode dev -Variants congestion -Threads 4 -ReferenceRun '.\runs\dev_20261016_160422'

# 14 天训练/3 天评测/50 轮，仅检查流程
powershell -ExecutionPolicy Bypass -File .\run_task1_experiments.ps1 -Mode smoke -Threads 4

.\.venv\Scripts\python.exe -m task1 ablate --help
.\.venv\Scripts\python.exe -m pytest tests -q
```

更改轮数等细项时使用直接 Python 命令，并去掉不再匹配的 `--reference-run`。
如需用新增模型生成正式预测，原 predict 命令的 `--models` 指向该组目录，
例如 `runs\ablation_dev_时间戳\models\multihop`；文件依然要与其他任务合并。
