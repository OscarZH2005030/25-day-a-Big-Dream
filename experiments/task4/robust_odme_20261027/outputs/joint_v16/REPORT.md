# v16：跨方向联合选择的校准可行性检查

读取v15相同固定场景的calibration四项分数，为十个方向各选26个候选之一，用二进制MILP最大化总体S_dev增量。对16种误差族分别要求S_od、S_link、S_dev、S_attr及综合分均不低于v11；比只要求总体平均不降更严格。

结果选择全部v11，离散策略的校准S_dev增益为0。MILP原始目标约1e-9的数值差异不算进展。此结论仅限该候选网格、这些校准场景和求解器容差，不能证明所有模型都无解。没有为了制造增益而放松其他误差族的限制。

此阶段没有新的非基准策略，因此未增加计数CV或路径推理；count_CV_not_yet_checked明确记录。v15冻结基准策略已另做20261209/10计数划分。没有拿校准可行性检查当作重尾/稀疏未见测试，也没有实际隐藏路径标签。原基准与默认v9入口保留，未提交Kaggle。

复现：--stage global-screen；前提是已存在v15 calibration_candidates.csv（包内包含，或先执行--stage bidirectional）。完整相同测试四项与总分见../joint_v15/all_versions_same_test.csv。

求解记录：

{
  "policy": {
    "D12_I405_N": "v11",
    "D12_I405_S": "v11",
    "D12_I5_N": "v11",
    "D12_I5_S": "v11",
    "D7_I10_E": "v11",
    "D7_I10_W": "v11",
    "D7_I210_E": "v11",
    "D7_I210_W": "v11",
    "D7_I405_N": "v11",
    "D7_I405_S": "v11"
  },
  "calibration_Sdev_gain": 0.0,
  "raw_MIP_objective_gain": 8.106291786985674e-10,
  "constraints": "for each of 16 error families, all four components plus total no worse on calibration; select one candidate per panel",
  "calibration_only_screen": true,
  "count_CV_not_yet_checked": true,
  "hidden_truth_available": false,
  "MIP_optimal": true,
  "MIP_gap": 0.0,
  "scope": "only this 26-candidate discrete grid, subject to numerical solver tolerances; not impossibility for all models",
  "promoted": false
}
