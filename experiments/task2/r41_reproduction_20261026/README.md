# Task2 R41 在本机复现（2026-10-26）

来源：朋友发布的 Release `task2-onset-best-20261026` 中的 `Phase367-R41.zip`
（SHA256 `67e211028d724f6f71bbb61b48fcb46748b1c999ced11d827dd0f5713764e1e8`，已校验），
模型代码与权重未修改。环境按包内 requirements.txt 固定：numpy 2.2.6、pandas 2.3.3、pyarrow 26.0.0、lightgbm 4.7.0、torch 2.9.1+cu128，Python 3.12。

| 项 | 结果 |
|---|---|
| `verify.py` | PASS，1167 个保存文件 |
| validation | 87,000 行，约 82 秒 |
| private | 87,000 行，约 127 秒 |
| 与包内 `predictions/queue-*.csv` | 键顺序和 `queue_pred` 逐行完全一致 |
| 官方模板 | 键完全匹配，无重复，值仅 0/1，8 个方向 × 10 窗口 |

文件：`r41_validation.csv`、`r41_private.csv`（SHA256 见 `SHA256SUMS.txt`）、两份运行日志。

分数口径：本地代理 IoU（新 367 窗口 综合 0.7907 / Onset 0.7215 / Ongoing 0.8599；旧 410 窗口 0.7507）来自朋友的报告，
由速度阈值构造标签，二月数据已反复用于开发，不是独立盲测；官方 `S_queue` 未知，需 Kaggle 验证提交。
本目录未做评分，也未提交 Kaggle。
