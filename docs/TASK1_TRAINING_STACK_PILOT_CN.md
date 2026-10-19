# Task 1 模型训练栈首轮试运行

## 本轮做了什么

- 在隔离的 Python 3.12 环境中，用官方训练数据对 `D12_I5_N` 启动了现有图 Transformer 训练入口。
- PyTorch 为 `2.7.0+cu128`，成功使用 NVIDIA RTX 4090 Laptop GPU；LightGBM、PyArrow 等依赖也能正常导入。
- 使用仓库已有的 51 项基础特征，加 51 项有效值标记，共 102 维输入；模型有 208,678 个参数，时序窗口 24，图节点分块上限 64，batch size 2。
- 按模型分支的日期边界缓存了 191 个日期；监督训练块为 168 天，内部验证为 22 天，诊断留出只取 2031-01-05 一天。
- 完成 1 个 epoch、4 次优化器更新。FD 与 conservation proxy 权重为 0，因此本轮优化目标只有重建损失。
- 训练曲线中的末步训练 loss 为 `0.065519`；22 天内部验证的 `S_state` 为 `0.906338`。

## 结果如何解释

这是一次训练链路试运行：数据读取、特征缓存、CUDA 前向与反向传播、参数更新、内部验证及 checkpoint 保存都已走通。模型权重、训练曲线和配置保存在本机忽略目录：

`outputs/task1_training_stack/smoke_d12_i5_n/D12_I5_N/`

它不是可比较的正式 baseline：只有 4 次更新，训练明显不充分；分数只覆盖一条走廊，且不是 Kaggle leaderboard 或 Task 1 全任务分数。

## 本轮遇到的问题

1. 第一次启动时，`--history` 指向了错误的上一级目录。模型尚未开始训练；修正为与 `--output` 相同的走廊目录后重跑。
2. 完成训练后，脚本对 2031-01-05 单日留出数据打分失败：此切片没有 R2 样本，但当前评分函数要求 R1、R2、R3 都有目标。因此未生成完整 `metrics.csv` 或有效留出总分。模型权重与内部验证记录已在失败前保存。
3. 项目既有文档说明 2031-01-05 至 2031-02-28 的本地留出期此前已被反复查看；这次也读取并尝试评估了 2031-01-05 标签。因此不能把它称作未触碰测试集，也不应拿本轮记录作为模型择优依据。

## 可复现命令

从仓库根目录的 PowerShell 执行；当前命令复现的是本轮小规模训练，输出使用新目录时应同步修改 `--history` 与 `--output`：

```powershell
.\.venv-task1-py312\Scripts\python.exe -u experiments\task1\full_dataset_training_v2\full_train_worker.py `
  --release-root data\kaggle_public `
  --panel D12_I5_N `
  --history outputs\task1_training_stack\smoke_d12_i5_n\D12_I5_N\history.npz `
  --output outputs\task1_training_stack\smoke_d12_i5_n\D12_I5_N `
  --train-start 2030-06-29 `
  --internal-val-start 2030-12-14 `
  --val-start 2031-01-05 `
  --val-end 2031-01-06 `
  --device cuda --epochs 1 --steps-per-epoch 4 --batch-size 2 `
  --window 24 --patch-nodes 64 --width 64 --heads 4 --layers 3 `
  --patience 1 --threads 4 --fd-weight 0 --conservation-weight 0
```

## 下一步

在做正式比较前，先与队友统一 Task 1 的训练、模型选择和留出日期边界；为完整留出区间生成包含 R1/R2/R3 的评估切片。随后再增加训练预算，并按同一切分完成 Transformer、LightGBM 与门控残差对照。现有门控残差流水线要求完整 Transformer 与 LightGBM 参考产物，单靠本轮 smoke run 不能启动其正式阶段。
