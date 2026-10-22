# 最新实验：T1／动态缺失六方向队列（2026-10-21）

这是当前正在进行的 Transformer 实验，面向 **Windows＋NVIDIA 显卡**的新电脑。完整训练源码与原电脑逐字节一致，新增入口只处理路径、来源参考表和队列启动。

三批依次运行 D7_I210_E/W、D7_I10_E/W、D12_I405_N/S；每方向 T1、A_dynamic、T0，共 **18 个神经模型**，另训练各方向共享 LightGBM。已完成 D7_I405_N/S、D12_I5_N/S 四方向使用随包提供的固定参考表。训练缓存会在新电脑自动建立。

## 1. 下载代码和官方数据

```powershell
git clone https://github.com/OscarZH2005030/25-day-a-Big-Dream.git
cd .\25-day-a-Big-Dream\experiments\task1_t1_dynamic_20261021
```

已有仓库时先在仓库根目录运行 `git pull --ff-only`，再进入上述实验目录。

自行下载比赛官方数据，将 **kaggle_public 文件夹**放成下面的结构。目录内同时存在代码包 `task1/` 和数据路径中的 `task1/`，两者不要互相覆盖。

```text
task1_t1_dynamic_20261021/
  task1/                         # Python 源码
  transformer_remaining/
  references/                    # 已校验的旧四方向固定结果表
  2026-ieee-big-data-traffic-flow-bench/
    kaggle_public/
      config/
      corridors/
      task1/
      ...
```

也可以沿用电脑上已有的数据目录，在所有运行命令中传 `-ReleaseRoot 'D:\数据目录\kaggle_public'`。数据与代码均支持放在包含空格或中文的路径下。数据、模型、缓存和运行输出不上传到 Git。

## 2. 建立环境

安装 Python **3.14**（参考电脑为 3.14.8），然后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_team.ps1
```

脚本建立一套 `.venv-grin`，安装锁定依赖与 **PyTorch 2.10.0+cu128**，检查 NVIDIA GPU、107 份源码及依赖的哈希、固定参考表和自动化检查。可用 `-PythonPath 'C:\路径\python.exe'` 指定 Python 3.14。不会安装或更改系统 CUDA Toolkit。

如已具备完全匹配的环境，可直接检查并在运行命令中传 `-PythonPath '已有环境\Scripts\python.exe'`：

```powershell
.\.venv-grin\Scripts\python.exe -B .\check_environment.py --device cuda
```

## 3. 先小检查，再正式跑

```powershell
# 三批六方向，18 个短模型，每模型 2 轮 × 4 步；输出明确标为工程检查
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode smoke -Device cuda -Output '.\runs\friend_smoke_001'

# 小检查通过后，启动正式三批队列
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -Device cuda -Output '.\runs\friend_formal_001'
```

每方向保留原 51 特征／102 输入、window=24、patch=64、width=64、3 层、batch=2、种子和优化器；最多 **60 轮 × 1024 步**，patience=8。更强显卡仍使用这套预算，以便保持可比性。

默认先做 I210，再做 I10，最后做 D12_I405，每批独立子进程串行运行。可加 `-Batch i210`、`-Batch i10` 或 `-Batch i405` 只跑一批；这种输出只报告所选方向。完整十路报告需要三批都在同一个顶层队列中完成。

如想先单独准备完整日期输入，可另起准备目录，随后正式队列会复用缓存：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -PrepareOnly -Output '.\runs\friend_prepare_001'
```

## 4. 暂停与续跑

在运行窗口按 **Ctrl+C**。续跑使用原来的顶层队列目录，保持相同模式、设备、数据路径、缓存路径和批次选择：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -Device cuda -Resume -Output '.\runs\friend_formal_001'
```

顶层目录有 `queue_config.json`，`batches/01_D7_I210` 等为子目录。已完成模型和批次会经校验复用，未完成 epoch 从上一个完整断点重做。准备目录、小检查目录和正式目录分别使用不同名称。

## 5. 输出与实验日志

每批保留 T0、T1、A_dynamic、LightGBM、原固定融合与替换 T1 后融合的预测、训练曲线、逐 regime／缺口／日期／拥堵／孤立节点诊断及来源清单。正式完成后自动追加本目录的 `实验日志.txt`，重复续跑不会重复记账。

三批都完成后输出：

- `ten_direction_comparison.csv`：五种共同方法的十方向／五家族宏平均及各对照差值。
- `ten_direction_metrics_by_panel_regime.csv`、`ten_direction_scores_by_panel.csv`、`ten_direction_scores_by_family.csv`。
- `ten_direction_daily_errors.csv`、`ten_direction_*_errors.csv`、配对日期重采样区间。
- `summary_sources.json`、`summary_integrity.json`、`COMPLETE.json`：来源与验收记录。

**十路表由本次新训练六方向＋固定历史四方向组成。** `references/` 的原始来源为 `transformer_training_dev_20261019_183939_881805` 和 `d12_i5_dynamic_dev_20261020_124730_964790`。导出表逐文件核对原完成清单，并检查日期、目标数、regime、孤立节点回退记录及逐日误差能否重建分数。原始完成标记和文件清单一并保留；新电脑不会重新验证旧四路的模型权重或逐目标预测，也不会重新训练旧四路。

三批工程小检查只生成 `smoke_selected_directions_comparison.csv` 和 `SMOKE_COMPLETE.json`，不混入历史正式分数、不写正式日志。若汇总或记日志被打断，可分别运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -SummaryOnly -Output '.\runs\friend_formal_001'
powershell -ExecutionPolicy Bypass -File .\run_team_remaining.ps1 -Mode dev -LogOnly -Output '.\runs\friend_formal_001'
```

## 固定实验协议

- 日期：28 天历史、168 天拟合、22 天内部选择、55 天留出；静态日期表随代码提供。
- 评分：速度 RMSE /25、每车道流量 RMSE /600，权重 0.54／0.46；按 regime→方向→家族等权聚合。
- 连通节点两通道均为 45% T0＋45% T1／A_dynamic＋10% LightGBM；孤立节点两通道精确回退到共享 LightGBM。
- 所有神经模型和内部产物冻结后才读取对应批次留出标签。动态缺失只在允许的训练观测上生成，输入同步重算。
- `0.936910312032` 是历史本地目标线。本地留出已反复参与开发，单种子结果和描述性日期区间不代表 Kaggle 隐藏测试成绩。

原始冻结计划见 [PLAN_TASK1_REMAINING_DYNAMIC.txt](PLAN_TASK1_REMAINING_DYNAMIC.txt)。`transformer_remaining/` 等包保留原源码；新电脑请使用本页的 `run_team_remaining.ps1`，它会加载便携参考适配器。`team_portable.py` 只替换旧四路参考的读取及校验方式，训练、评分、固定融合和回退沿用原实现。

## 本次发布检查

Python 3.14.8／锁定依赖下，107 份源码和依赖的哈希校验通过；**149 项检查通过，3 项 CUDA 用例在 CPU 自检中跳过**。覆盖新电脑空缓存配置、原训练采样与恢复、固定融合和孤立回退、参考表篡改拒绝，以及真实旧四路表与新六路模拟结果的十路聚合。

原有三项 CUDA 参数逐位相等测试曾出现浮点差异；冻结训练代码保持原样，安装脚本明确在 CPU 上做这些单元检查。朋友电脑的实际 GPU 工程运行请先执行上面的 `-Mode smoke`，通过后再启动正式训练。发布副本没有额外执行正式训练。
