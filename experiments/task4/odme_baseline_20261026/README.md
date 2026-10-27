# Task4 官方基线（正则化非负最小二乘）本机复现，2026-10-26

方法：`trafficflowbench-public/src/task4/build_task4_odme_artifacts.py`，min ||A f − c||² + λ||f − b||²，f ≥ 0，λ = 0.05（官方默认），未调参。
范围：validation 与 private，各 35,354 行，10 个方向；无负值、无非有限值、无重复键，行数与 `submission_key.csv` 的 odme 行一致。

| 项 | 结果 |
|---|---|
| validation 整体 S_link | 0.964566 |
| 最高方向 | D12_I405_S 0.9878 |
| 最低方向 | D7_I210_W 0.9455 |

按家族：D12_I405 0.983、D7_I405 0.968、D7_I10 0.962、D7_I210 0.958、D12_I5 0.952。逐方向见 `task4_link_fit_validation.csv`。

口径：S_link 只占 Task4 的 25%，是本地唯一能算的一项。S_od、S_dev、S_attr 需要未公开的真实路径流量，只能提交 Kaggle 后在榜上看到。
基线自身生成的参考文件不能用于本地打分（同一道题的解，会得虚高满分），故未收录。本目录未提交 Kaggle，也不是改进模型。
文件：`odme_validation.csv`、`odme_private.csv`（SHA256 见 `SHA256SUMS.txt`），列为 panel, departure_time, path_id, origin_zone, destination_zone, path_flow。
