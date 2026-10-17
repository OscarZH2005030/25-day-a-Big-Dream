# Task 1 训练与复现

先读 [实验说明与结果](../../docs/TASK1_GATED_RESIDUAL.md)。下列是复现命令；整理上传不会执行它们。
必须在独立 checkout、独立输出目录运行，不能用这些副本续跑原机器正在运行的实验。
此流水线目前需要从数据重建原 Transformer 缓存、完成其十个模型，再完成公平 LightGBM，然后运行门控残差；不是下载仓库后只跑最后一步即可。
本地旧缓存 manifest 含数据绝对路径与 mtime，不能简单跨机器搬移；应在目标机器重建，不能关闭校验。

从仓库根目录启动 PowerShell。先准备 Python 3.11 和与本机驱动匹配的 PyTorch，记录环境为 2.7.0+cu128。

```powershell
python -m pip install -r experiments/task1/requirements.txt
$repoRoot = (Get-Location).Path
$taskRoot = Join-Path $repoRoot 'experiments/task1'
$dataRoot = Join-Path $repoRoot 'data/kaggle_public'
$tfSource = Join-Path $taskRoot 'full_dataset_training_v2'
$tfRun = Join-Path $tfSource 'runs/all_corridors_time80_20'
$lgbRun = Join-Path $taskRoot 'lightgbm_time80_20/runs/all_corridors_time80_20'
$gatedRun = Join-Path $taskRoot 'lgb_gated_residual_v1/runs/all_corridors_time80_20'
```

数据解压为 `data/kaggle_public/`；原 Transformer 使用 CLI 的 release-root，因此也可把 `$dataRoot` 改为本机数据根目录。

1. 原图 Transformer（会训练、生成日期缓存并评估）：

```powershell
python -u "$tfSource/train_all_corridors.py" --release-root "$dataRoot" --output "$tfRun" --resume
```

2. 完成全部十条后训练公平 LightGBM（CPU，所有模板训练目标）：

```powershell
python -u "$taskRoot/lightgbm_time80_20/train_lightgbm.py" --source "$tfSource" --reference-run "$tfRun" --output "$lgbRun" --resume
```

3. 只做新模型前置检查；不拟合树、不初始化 CUDA：

```powershell
python -u "$taskRoot/lgb_gated_residual_v1/run_pipeline.py" --stage preflight --reference-run "$tfRun" --final-lgb "$lgbRun" --output "$gatedRun"
```

4. 显式启动交叉拟合→残差训练→留出评估，十条独立模型顺序执行：

```powershell
python -u "$taskRoot/lgb_gated_residual_v1/run_pipeline.py" --stage all --device cuda --reference-run "$tfRun" --final-lgb "$lgbRun" --output "$gatedRun" --resume
```

`--stage` 默认是 preflight。中断后用相同源码、参数与输出目录加 `--resume`；轮次检查点保存模型、优化器、调度器及 RNG。
队列与工作目录有独占锁；不要同时开启两份。每轮 1024 步，最多 60 轮，内部验证早停 8 轮。
源码、输入签名和计划必须匹配，复制后的文件不要对原训练目录使用 `--resume`。

可选 GPU 活动伴随脚本（会启动上述全部流水线，不能与第 4 步同时启动）：

```powershell
python -u "$taskRoot/run_gated_residual_with_gpu_activity.py"
```

伴随脚本在独立进程持续执行小 CUDA 运算，并阻止系统自动睡眠，会额外消耗 GPU 资源／电力；不能保证修复驱动问题。
需要自定义路径时使用其 `--help` 查看转发参数。只需训练时可直接使用第 4 步。

查看状态及日志：

```powershell
Get-Content "$gatedRun/status.json"
Get-Content "$gatedRun/D12_I405_N/console.log" -Tail 30 -Wait
```

crossfit 的状态时间戳可能滞后，结合日志判断。完成以走廊 COMPLETE.json 为准，不能只看 running 字段。

无训练检查：`python experiments/task1/verify_results.py`（仅标准库）核对文件哈希、评分公式、聚合与独立 SSE 重建。
`python experiments/task1/lgb_gated_residual_v1/self_checks.py` 是已有合成 CPU 自检，会 CPU 前向／反向，不拟合树、不执行优化器更新、不初始化 CUDA；写入该副本下的 self_checks.json。
整理时只执行轻量语法、路径与结果检查；已有 CPU 自检记录另存于 docs/results。

分析脚本：`internal_gap_switch/evaluate_switch.py` 用 CPU 推理生成内部预测；`internal_fallback_switch/evaluate_fallback.py` 读取这些预测测试回退；`holdout_isolated_fallback/evaluate_holdout.py` 在原模型留出预测上应用固定孤立回退；`remaining_four_diagnosis/analyze_remaining.py` 按通道解释剩余误差。
这些脚本会生成各自 results/，不随仓库分发。内部回退默认数据目录为仓库 data/kaggle_public；其他位置需修改其导出副本 RELEASE 常量。
留出脚本引用随附历史 topology_audit.csv 进行一致性核对；历史独立 SSE 证据只适用于随附模型结果，重训后的独立证据需重新生成。
