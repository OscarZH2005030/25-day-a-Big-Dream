# v14：计数偏差判断与严格四项条件

本轮仍使用完全相同的480场景与随机种子，v11参考逐项一致<1e-10。v12稀疏调整、v13道路/目的地等式约束结果全部保留。v14增加只读取log(c/(A*b))的稳健判断：位置取中位数、尺度取1.4826*MAD（最低0.02），共同偏移信号超过2.5个尺度或单侧尾部偏移超过2个尺度才逐渐加权。系数固定，不读取场景名、真值、行位置或已知注入误差幅度。它是针对已见问题开发的假设，不是未见盲测，不能用来声称真实比赛提升。

新调整仍使用稀疏LP最小化偏离，严格保持v11每个计数链接的加载和每个目的地的绝对总量，因此S_link、S_attr在数值精度内相同。S_od、S_dev仍需检验。选择要求校准总体及局部漏计、整体计数偏高各族四项和总分均不降，计数CV平均误差不增。在通过者中最大化目标族S_dev。所有通过/失败条件见selection_checks.csv。只复核冻结策略；没有跑所有失败候选的重尾/稀疏LP。

前轮已经观察过测试场景。本轮选模只读calibration及20261201/02计数划分；20261203/04作复核。策略冻结后新增20261207/08两划分，v13/v14使用同样的划分便于对比；仍然是同一train计数重复划分，不是独立真实路径标签。

全版本同测试四项与总分：

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
  calibration      v14 0.507200 0.997369 0.654321 0.905824          0.711604
   evaluation      v14 0.496551 0.997219 0.661939 0.900791          0.707162
sparse_stress      v14 0.443977 0.996590 0.649332 0.887418          0.679450

两个目标误差族的冻结策略：

        phase             scenario method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
   evaluation   count_systematic30    v11 0.802329 0.999983 0.433370 0.960533          0.820129
   evaluation   count_systematic30    v14 0.802329 0.999983 0.433370 0.960533          0.820129
   evaluation localized_undercount    v11 0.537901 0.997211 0.874723 0.914612          0.759759
   evaluation localized_undercount    v14 0.537901 0.997211 0.874723 0.914612          0.759759
sparse_stress   count_systematic30    v11 0.678301 0.999848 0.184525 0.953896          0.725960
sparse_stress   count_systematic30    v14 0.678301 0.999848 0.184525 0.953896          0.725960
sparse_stress localized_undercount    v11 0.415540 0.997857 0.601242 0.880375          0.658700
sparse_stress localized_undercount    v14 0.415540 0.997857 0.601242 0.880375          0.658700

新增计数划分：

    seed method     NMAE
20261207    v11 0.032207
20261207    v14 0.032207
20261208    v11 0.041640
20261208    v14 0.041640

校准目标S_dev最高但尚未通过联合条件的候选（仅用于显示权衡，不推荐部署）：

experiment                                         label     candidate     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
       v12                                     reference           v11 0.708909 0.999093 0.717742 0.945560          0.818278
       v12 target_Sdev_best_before_guards_NOT_deployable elastic1_mix1 0.717608 0.999130 0.816928 0.944821          0.836968
       v13                                     reference           v11 0.708909 0.999093 0.717742 0.945560          0.818278
       v13 target_Sdev_best_before_guards_NOT_deployable    invariant1 0.706711 0.999093 0.733103 0.945560          0.819593
       v14                                     reference           v11 0.708909 0.999093 0.717742 0.945560          0.818278
       v14 target_Sdev_best_before_guards_NOT_deployable      guarded1 0.717667 0.999093 0.724911 0.945560          0.823294

冻结策略变动方向：
{}

评估条件：
[
  {
    "phase": "evaluation",
    "scenario": "all",
    "all_components_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "evaluation",
    "scenario": "localized_undercount",
    "all_components_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "evaluation",
    "scenario": "count_systematic30",
    "all_components_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "sparse_stress",
    "scenario": "all",
    "all_components_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "sparse_stress",
    "scenario": "localized_undercount",
    "all_components_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  },
  {
    "phase": "sparse_stress",
    "scenario": "count_systematic30",
    "all_components_and_total_nondecreasing": true,
    "delta_S_od": 0.0,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0
  }
]

三类约束不能凭空恢复不可识别的路径真值。减少偏离可能帮助计数偏差场景，却伤害本来就低估真实变化量的场景。未通过条件就保持v11，绝不把单项高分称作联合提升。actual_four_components.csv只填公开计数可计算的S_link，其他三项及实际总分为空。

未覆盖原始正式提交，未改变默认v9入口，未提交Kaggle。--stage guarded重跑v14，--stage guarded-infer复现冻结策略与两组计数复核。v12/v13入口为bias/invariant，失败实验与代码均保留。
