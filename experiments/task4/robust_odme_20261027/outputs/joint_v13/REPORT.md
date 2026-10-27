# v13：锁定道路和目的地，再优化偏离量

同一套v10/v11的480场景，真值、先验、计数、随机种子及误差强度不变。旧v11逐项核对误差<1e-10；all_versions_same_test.csv并列官方、v9、v10、v11、v13。数据此前已查看，属于重复敏感性分析，不是独立路径盲测。

v12稀疏调整方法全部未通过联合条件；本轮进一步用稀疏LP最小化sum|x-b|，非负约束下严格保持A*x=A*f_v11和每个目的地的绝对总量不变。与v11按0.1/0.25/0.5/1混合也保持这些约束。因此计数拟合和S_attr在数值精度内保持，而S_od和S_dev仍必须通过实验验证；减小偏离不必然提高S_dev。

只用calibration和20261201/02计数划分选择。要求总体及两个目标误差族的四项和综合均不降、计数CV平均不增且最差种子比值<=1.01，通过后最大化两个目标族的S_dev。未通过人工校准条件的候选不再做额外LP计数验证，real_count_cv.csv因此只含实际检验过的候选。重尾和稀疏阶段只评估冻结策略，未评估全部失败候选。20261207/08为策略冻结后新加的计数划分，仍复用同一比赛train计数，不是独立真实路径标签。

全量四项与综合：

        phase   method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
  calibration      v11 0.507200 0.997369 0.654321 0.905824          0.711604
  calibration official 0.498240 0.997363 0.654881 0.904470          0.707452
  calibration      v10 0.500402 0.997364 0.654961 0.904863          0.708496
  calibration       v9 0.504455 0.997368 0.654187 0.905374          0.710281
   evaluation      v11 0.496551 0.997219 0.661939 0.900791          0.707162
   evaluation official 0.488663 0.997212 0.662204 0.899558          0.703466
   evaluation      v10 0.490484 0.997214 0.662038 0.899907          0.704313
   evaluation       v9 0.494196 0.997217 0.662033 0.900376          0.706054
sparse_stress      v11 0.443977 0.996590 0.649332 0.887418          0.679450
sparse_stress official 0.435081 0.996579 0.646841 0.885471          0.674778
sparse_stress      v10 0.437919 0.996581 0.648318 0.885990          0.676355
sparse_stress       v9 0.440899 0.996587 0.648098 0.886823          0.677789
  calibration      v13 0.507200 0.997369 0.654321 0.905824          0.711604
   evaluation      v13 0.496551 0.997219 0.661939 0.900791          0.707162
sparse_stress      v13 0.443977 0.996590 0.649332 0.887418          0.679450

两个目标误差的四项与综合：

        phase             scenario method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
   evaluation   count_systematic30    v11 0.802329 0.999983 0.433370 0.960533          0.820129
   evaluation   count_systematic30    v13 0.802329 0.999983 0.433370 0.960533          0.820129
   evaluation localized_undercount    v11 0.537901 0.997211 0.874723 0.914612          0.759759
   evaluation localized_undercount    v13 0.537901 0.997211 0.874723 0.914612          0.759759
sparse_stress   count_systematic30    v11 0.678301 0.999848 0.184525 0.953896          0.725960
sparse_stress   count_systematic30    v13 0.678301 0.999848 0.184525 0.953896          0.725960
sparse_stress localized_undercount    v11 0.415540 0.997857 0.601242 0.880375          0.658700
sparse_stress localized_undercount    v13 0.415540 0.997857 0.601242 0.880375          0.658700

新增计数验证：

    seed method     NMAE
20261207    v11 0.032207
20261207    v13 0.032207
20261208    v11 0.041640
20261208    v13 0.041640

通过筛选的改动方向：
{}

全部验收结果：
[
  {
    "phase": "evaluation",
    "scenario": "all",
    "all_four_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "evaluation",
    "scenario": "localized_undercount",
    "all_four_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "evaluation",
    "scenario": "count_systematic30",
    "all_four_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "sparse_stress",
    "scenario": "all",
    "all_four_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "sparse_stress",
    "scenario": "localized_undercount",
    "all_four_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "sparse_stress",
    "scenario": "count_systematic30",
    "all_four_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  }
]

invariance_certificates.csv与actual_invariance_certificates.csv记录等式误差。实际数据只能计算公开计数S_link；S_od、S_dev、S_attr及实际总分未知，actual_four_components.csv明确留空。没有把公开参考解当作隐藏真值，没有新增真实路径标签，没有新下载五分钟数据。

未覆盖原始正式提交，默认v9探索推理仍保留。复现：--stage bias运行v12，--stage invariant运行v13及冻结推理/新增计数复核，--stage invariant-infer只执行冻结策略推理与计数复核。未提交Kaggle。
