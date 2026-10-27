# Task4 v11：固定测试，联合优化四项

本轮完全固定v10的480个场景（十方向、16种误差、三种形状），不改变真值、误差幅度或随机种子。官方基准、v9、v10在新旧表中逐项核对，最大差异小于1e-10。每个场景的f/b/c哈希保存在fixed_case_hashes.csv。

这些仍然是人工路径真值敏感性实验，不是比赛真实分数。该测试以前已经查看过，本轮新方法开发也受到既有发现启发，因此评估不属于未见盲测；即使程序选参只读calibration，也不能把重测结果当成独立泛化证明。真实计数也复用同一train，与真实路径标签不同。

新方法：沿KL修正方向适度外推、非负截断后按原计数重新投影；按共同来源/目的地平滑KL的对数修正再投影。共26个候选。所有预测函数只读取网络、公开计数和弱先验，不读取人工场景名或真值。

选择：以v9的各方向策略为对照，仅用calibration及20261201/02计数划分。要求四项及综合均不下降、真实计数平均误差不增、任一种子比值<=1.01、最差误差族综合损失<=0.005，在通过的候选中最大化S_dev。条件约束的是校准均值，并不保证所有场景都变好。20261203/04用于计数复核，但这些划分此前也已查看，不声称独立盲测。

四项与综合分同表（人工真值）：

        phase        method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
  calibration joint_dev_v11 0.507200 0.997369 0.654321 0.905824          0.711604
  calibration      official 0.498240 0.997363 0.654881 0.904470          0.707452
  calibration  previous_v10 0.500402 0.997364 0.654961 0.904863          0.708496
  calibration   previous_v9 0.504455 0.997368 0.654187 0.905374          0.710281
   evaluation joint_dev_v11 0.496551 0.997219 0.661939 0.900791          0.707162
   evaluation      official 0.488663 0.997212 0.662204 0.899558          0.703466
   evaluation  previous_v10 0.490484 0.997214 0.662038 0.899907          0.704313
   evaluation   previous_v9 0.494196 0.997217 0.662033 0.900376          0.706054
sparse_stress joint_dev_v11 0.443977 0.996590 0.649332 0.887418          0.679450
sparse_stress      official 0.435081 0.996579 0.646841 0.885471          0.674778
sparse_stress  previous_v10 0.437919 0.996581 0.648318 0.885990          0.676355
sparse_stress   previous_v9 0.440899 0.996587 0.648098 0.886823          0.677789

重尾评估中S_dev下降最大的五种误差（各项相对v9的差值）：

                          S_od        S_link     S_dev    S_attr  synthetic_S_ODME
scenario
localized_undercount  0.001905  8.368769e-07 -0.002400  0.000128          0.000517
count_systematic30    0.000957  1.913866e-07 -0.002249  0.000573          0.000179
origin_bias           0.006058  1.527319e-06 -0.001519  0.000625          0.002592
rounded_counts        0.002082  4.046750e-07 -0.001055  0.000333          0.000829
destination_bias      0.005184  1.888350e-06 -0.000874  0.000515          0.002279

比赛公开计数留出误差（NMAE，越低越好）：

    seed        method     NMAE
20261201 joint_dev_v11 0.034341
20261201      official 0.035126
20261201  previous_v10 0.035012
20261201   previous_v9 0.034494
20261202 joint_dev_v11 0.037270
20261202      official 0.037447
20261202  previous_v10 0.037261
20261202   previous_v9 0.037447
20261203 joint_dev_v11 0.038076
20261203      official 0.038845
20261203  previous_v10 0.038697
20261203   previous_v9 0.038259
20261204 joint_dev_v11 0.035427
20261204      official 0.035958
20261204  previous_v10 0.035828
20261204   previous_v9 0.035580

冻结策略后新增加的两个划分（同一train计数重复划分，非独立路径真值）：

    seed        method     NMAE
20261205 joint_dev_v11 0.030952
20261205      official 0.031690
20261205  previous_v10 0.031543
20261205   previous_v9 0.031137
20261206 joint_dev_v11 0.038785
20261206      official 0.039443
20261206  previous_v10 0.039327
20261206   previous_v9 0.038941

固定策略：

{
  "D12_I405_N": "official",
  "D12_I405_S": "official",
  "D12_I5_N": "official",
  "D12_I5_S": "official",
  "D7_I10_E": "official",
  "D7_I10_W": "official",
  "D7_I210_E": "official",
  "D7_I210_W": "scaled_amp1.1_mix0.25",
  "D7_I405_N": "official",
  "D7_I405_S": "scaled_0.1"
}

相对v9复核：

[
  {
    "phase": "evaluation",
    "delta_S_od": 0.002354240028604071,
    "delta_S_link": 1.821168455817812e-06,
    "delta_S_dev": -9.403626484538741e-05,
    "delta_S_attr": 0.00041451538437187185,
    "delta_synthetic_S_ODME": 0.0011079351729147424,
    "all_components_non_decreasing": false
  },
  {
    "phase": "sparse_stress",
    "delta_S_od": 0.00307807560095541,
    "delta_S_link": 2.38635613347693e-06,
    "delta_S_dev": 0.0012339654157887958,
    "delta_S_attr": 0.0005949070293023162,
    "delta_synthetic_S_ODME": 0.0016600614762268817,
    "all_components_non_decreasing": true
  }
]

真实数据能做什么：比赛发布的计数可检验道路总量拟合和留出计数预测；先前下载的匿名五分钟流量可用于现实波动场景参考，缺少同时间的路径标签且覆盖有限。本轮不新增五分钟数据源，不把历史流量当作本次比赛的路径真值。

比赛方不是没有另外三项。其docs/RUN_LOCAL.md明确指出S_od、S_dev、S_attr比较主办方保留的路径流量，只在排行榜评分；不能把公开计数构造的参考解当作隐藏真值。actual_four_components.csv只填可计算的真实S_link，其余保持未知。

原始正式提交没有覆盖，没有提交Kaggle。默认复现入口仍保留v9；v11独立入口使用joint-dev与joint-dev-infer。原始预测、失败方法与逐场景结果全部保留。
