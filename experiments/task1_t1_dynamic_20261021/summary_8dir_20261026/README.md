# T1-dynamic 八方向汇总（2026-10-26，Oscar）

范围：已完成的 8 个方向，来源为 Jayden 旧四方向参考表（D7_I405、D12_I5）和 Oscar 在 ROG 上补跑的 D7_I10、D12_I405
（`runs/oscar_handoff_i10_formal_20261022`、`runs/oscar_handoff_i405_formal_20261022`，均有 COMPLETE.json）。

**缺 D7_I210_E/W**：结果仍在 Jayden 的 Dell 上，尚未推送。所以这不是十方向总分，**不能和历史 0.936910312032 直接比较**。

聚合：每方向按 R1/R2/R3 等权，再对同一走廊家族内方向等权，再对 4 个家族等权（与批次内 holdout_comparison_methods.csv 的数值核对一致）。
LightGBM 在 D7_I405 参考表中没有，故不进总分，只在可用方向展示。

| 方法 | S_state（8方向/4家族） | 对 Current_fusion | 对 T1 |
|---|---|---|---|
| A_dynamic_fixed | 0.939788 | +0.000237 | +0.000880 |
| Current_fusion | 0.939551 | 0 | +0.000643 |
| A_dynamic | 0.939408 | -0.000143 | +0.000500 |
| T1 | 0.938908 | -0.000643 | 0 |
| T0 | 0.938737 | -0.000814 | -0.000171 |

详见 `Task1_T1dynamic_8dir_summary_20261026.xlsx` 与各 CSV。属本地留出诊断，不等于 Kaggle 隐藏测试成绩。
