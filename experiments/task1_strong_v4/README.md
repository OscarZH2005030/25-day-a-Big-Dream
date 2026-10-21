# IEEE TrafficFlowBench 当前最佳模型：Strong v4

截至 2026-10-20，本包保存当前本地验证最佳的完整模型、依赖、代码、实验记录和已校验提交。原始工作区中的 v1–v3 均未覆盖。

## 先用哪个文件

- `submission.csv`：可直接用于比赛提交的完整候选，6,980,503 行；没有自动上传 Kaggle。
- `MODEL_CARD.json`：当前模型及验证结果。
- `predict_best.py`：可迁移的原始数据推理入口，从本目录加载所有权重和统计，不依赖旧 C 盘模型路径。
- `outputs/IEEE-Strong-v4/更大模型实测报告.txt`：本轮详细报告。
- `MANIFEST.json`、`verify_package.py`：完整文件指纹清单与校验入口。

## 当前结果与模型

| 相同 55 天本地留出数据 | v3 | 当前 v4 |
|---|---:|---:|
| S_state，越高越好 | 0.939851048 | **0.939898101** |
| 流量 RMSE，辆/小时/车道 | 34.439550 | **34.320188** |
| 速度 RMSE，km/h | 1.669400 | 1.669400 |

三个重点方向的主干从 11,388,974 参数扩为 43,855,662 参数（3.85 倍），宽度 512、10 个图/SSM 块、16 个全天注意力头；保留 288 时间点与完整道路站点。其余七方向沿用 v3。速度、排队和 ODME 输出保持原值。

三个扩容方向：D7_I405_N、D12_I5_N、D7_I210_E。其原有流量修正仍保留，部署表达式是 `v3_flow + alpha * (wide_flow - v2_raw_flow)`，三个方向的冻结 alpha 都是 1。

**这是多轮开发使用过的本地留出集结果，不是分类准确率或 Kaggle 隐藏测试分数。** 本轮只有一个种子、60 次继续训练更新，也没有等额追加训练的小模型对照，不能把收益全部归因于参数量增加。

## 为什么包含 27 份权重

当前模型包含分方向主模型、之前版本的校准与回退、局部流量专家和新扩容主干。只拿三个 v4 `.pt` 无法恢复完整预测。

- v1：十方向校准主模型、图结构和原始特征历史统计。
- v2：十方向继续训练主模型；D7_I405_N 的局部流量专家；三个方向的拟合期历史统计。
- v3：三个流量残差专家。
- v4：三个 43,855,662 参数主干。

完整包同时包含所有实际选用的权重、图、校准、历史统计、源代码、训练协议、日志、评分和逐方向结果。`SOURCE_COPY_MANIFEST.json` 记录来自原工作区的 849 份文件及拷贝核验结果。

## 在新目录运行

建议 Python 3.12 和 NVIDIA CUDA GPU。实验使用 RTX 3080 Laptop 16GB、PyTorch 2.9.1+cu126。先安装匹配的 CUDA PyTorch，再安装其余依赖：

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python verify_package.py
```

原始比赛数据需要自行从 Kaggle 下载。`--data` 必须指向包含 `submission_key.csv`、`corridors/` 和 `task1/` 的 `kaggle_public` 文件夹。

先重建 2031-03-01 的十方向预测，与包内保存的当前最佳预测比较：

```powershell
python predict_best.py --data "E:\data\kaggle_public" --out "E:\ieee-check" --smoke-reference
```

重建完整候选（会重新推理所有方向，已有提交不会覆盖）：

```powershell
python predict_best.py --data "E:\data\kaggle_public" --out "E:\ieee-new-submission"
```

完整运行会重新生成 Task 1 状态，复用包内已冻结的 Task 2/4 辅助预测，再按官方键合并并校验。它不是重新训练 Task 2/4。

## 代码、数据与历史记录的边界

`outputs/` 保留 v1–v4 的研究代码、配置和历史日志；历史文件中的旧绝对路径是审计记录。跨目录推理应使用本目录的 `predict_best.py`。继续训练的历史脚本需要重新生成特征/标签缓存并调整缓存路径，本包不承诺无需原数据即可重新训练整个研究过程。

没有包含原始 Kaggle 数据、标签缓存、巨大的中间特征、Python 虚拟环境或任何登录凭据。拟合期学习得到的历史统计作为模型资产包含。

## GitHub 存放方式

仓库中的本实验目录保存代码、配置、实验结果和文件清单；完整模型权重及提交位于同一仓库的 GitHub Release，避免普通 Git 大文件限制。请下载 Release 的 `IEEE-Best-v4-20261020.zip`，解压后运行上述命令。`download_model.py` 会检查发布的压缩包 SHA256 后解压。

目标 Release：<https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/ieee-strong-v4-20261020>

GPU 训练/内部验证约 30.3 分钟，功耗采样峰值 147.4 W、最高温度 72°C；该功耗是 GPU 功耗，不是整机功耗。
