# v17：真实OD/出行调查数据搜索、接入与同测试复核

## 先更正验证数据性质

本地比赛包的synthetic_release_v1.json明确声明不含历史交通观测行。此前“真实计数CV”应更正为“比赛公开合成计数留出验证”。历史数值保留，证据性质不再混用。详见DATA_VALIDATION_CORRECTION.md。真实外部LODES/NHTS与比赛合成计数是不同来源。本轮也没有竞赛隐藏路径标签。

## 实际取得的真实来源

- Census LODES8 2023：匿名下载California primary-job OD与2020 Census块坐标。两个CSV解压后的SHA256均与官方清单一致。区域限定为居住地、工作地都在Los Angeles/Orange两县；这是就业行政统计关联，不是车辆轨迹或PM车辆数。
- FHWA 2017 NHTS：匿名下载完整官方公共CSV ZIP。按HHSTATE=CA、HH_CBSA=31080、工作日TRAVDAY=2..6、DRVR_FLG=1及官方POV模式3/4/5/6/8/9/18筛选。它是洛杉矶都市区居民的出行调查，不能把家庭居住区当作每条出行的地理范围。PM按15:00至19:00前出发。WTTRDFIN用于聚合，未发布个人/家庭行记录。
- 2021 NextGen California OD addendum：官方公共目录提供细分OD与地理区文件，但年度/月度包为9.9/7.5 GB，本轮未取得其主体数据，不声称已验证下载或拿它做实验。
- NGSIM公开元数据包含车辆轨迹字段，但LA部分是US101/Lankershim，不是本次目标路线的路径标签。Metro I405公开报告包含LOCUS聚合图，未找到匿名原始轨迹表。DLSim ieee_sample虽有节点坐标，但同名链接与当前包长度不一致，不认定为可靠映射。全部核查结果见source_research_catalog.json。

真实数据清洗统计：

{
  "real_LA_Orange_census_blocks": 118360,
  "LODES_CA_rows_scanned": 14634271,
  "LODES_LA_OC_OD_pairs": 4315198,
  "LODES_LA_OC_primary_jobs": 4645915,
  "LODES_missing_coordinate_pairs": 0,
  "NHTS_national_trip_records_scanned": 923572,
  "NHTS_LA_OC_weekday_vehicle_trips": 10657,
  "NHTS_LA_OC_households": 1854,
  "NHTS_LA_OC_weekday_vehicle_records": 10657,
  "NHTS_LA_OC_weekday_vehicle_households": 1854,
  "NHTS_LA_OC_weekday_vehicle_weighted_mean_km": 13.494808973235813,
  "NHTS_LA_OC_PM_vehicle_records": 3114,
  "NHTS_LA_OC_PM_vehicle_households": 1392,
  "NHTS_LA_OC_PM_vehicle_weighted_mean_km": 13.306599526220872,
  "LODES_is": "annual employer/worker residence-workplace linkage; not observed vehicle trips, routes or PM departures",
  "NHTS_is": "2017 sampled household driver trips; survey-weighted lengths, no target-zone geographic crosswalk",
  "competition_geographic_zone_matches": 0,
  "competition_counts_are": "synthetic released counts, not historical measured traffic rows",
  "transfer_assumption": "trip lengths or commuter straight-line distances can only provide a weak regional length prior; corridor segment length is not full trip length"
}

## 为什么没有直接把真实OD贴到Z0/Z1

没有验证到发布区域与Census/TAZ/坐标的对应表。样本网络中重名链接不足以建立对应关系。当前每个OD只有一条候选路径，因此外部真实OD地理匹配比车道级轨迹更关键。另有大量路径缺少完整链接长度；只对长度完整可核实的路径施加外部先验，不补造缺失长度。方向中完整长度路径少于100条时完全回退v11。

匹配/长度审计：

     panel  paths  unique_OD_pairs  complete_length_paths  missing_length_paths  geographic_csv_files  verified_geographic_OD_zone_matches
D12_I405_N    903              903                    285                   618                   NaN                                    0
D12_I405_S    630              630                     14                   616                   NaN                                    0
  D12_I5_N   4278             4278                   4278                     0                   NaN                                    0
  D12_I5_S   5050             5050                     38                  5012                   NaN                                    0
  D7_I10_E   4278             4278                    347                  3931                   NaN                                    0
  D7_I10_W   5050             5050                     42                  5008                   NaN                                    0
 D7_I210_E   2628             2628                   2086                   542                   NaN                                    0
 D7_I210_W   3321             3321                     18                  3303                   NaN                                    0
 D7_I405_N   4560             4560                   2777                  1783                   NaN                                    0
 D7_I405_S   4656             4656                     16                  4640                   NaN                                    0

## 实验设置与结果

仍用完全相同的480个人工场景、随机种子、弱先验误差和计数误差。v11四项逐项核对误差<1e-10。外部真实数据只用于聚合长度先验：NHTS PM生存函数、PM密度、工作日生存函数及LODES直线距离×1.3的生存函数（1.3仅为路网绕行假设，不是实测路线长度）。整段走廊路径也不等于一次完整出行，迁移本身需要验证。

在长度已知的路径子集内，按来源分组分配外部长度权重，再投影到与v11相同的全部计数链接加载和每个目的地总量。因此S_link与S_attr在数值精度内保持。外部强度0.05/0.15/0.30，共12候选加v11。只读calibration与前两计数划分选择，约束总体及两目标族四项/总分不降；全部方向回退v11。

另外单列一个诊断：只用全局calibration均值选出5%的NHTS工作日长度先验，并在查看重尾/稀疏结果前写出冻结选择文件。它没有执行原来的逐方向/目标族选择条件，所以明确标为诊断，不冒充通过条件的方案。六组公开合成计数划分均单列。没有据诊断结果反向改参、未提交其全量预测。

严格策略四项与综合（人工真值）：

        phase          method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
  calibration external_od_v17 0.507200 0.997369 0.654321 0.905824          0.711604
  calibration             v11 0.507200 0.997369 0.654321 0.905824          0.711604
   evaluation external_od_v17 0.496551 0.997219 0.661939 0.900791          0.707162
   evaluation             v11 0.496551 0.997219 0.661939 0.900791          0.707162
sparse_stress external_od_v17 0.443977 0.996590 0.649332 0.887418          0.679450
sparse_stress             v11 0.443977 0.996590 0.649332 0.887418          0.679450

诊断候选四项与综合（人工真值；拒绝替换）：

        phase                     method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
  calibration global_external_diagnostic 0.507970 0.997369 0.661847 0.905824          0.713080
  calibration                        v11 0.507200 0.997369 0.654321 0.905824          0.711604
   evaluation global_external_diagnostic 0.496294 0.997219 0.665173 0.900791          0.707532
   evaluation                        v11 0.496551 0.997219 0.661939 0.900791          0.707162
sparse_stress global_external_diagnostic 0.438198 0.996590 0.620313 0.887418          0.672496
sparse_stress                        v11 0.443977 0.996590 0.649332 0.887418          0.679450

诊断候选的公开合成计数留出误差（不是实测交通验证）：

    seed                     method     NMAE
20261201 global_external_diagnostic 0.034263
20261201                        v11 0.034341
20261202 global_external_diagnostic 0.037184
20261202                        v11 0.037270
20261203 global_external_diagnostic 0.038012
20261203                        v11 0.038076
20261204 global_external_diagnostic 0.035385
20261204                        v11 0.035427
20261211 global_external_diagnostic 0.033260
20261211                        v11 0.033305
20261212 global_external_diagnostic 0.034739
20261212                        v11 0.034817

是否通过联合条件：
[
  {
    "phase": "evaluation",
    "scenario": "overall",
    "all_components_nondecreasing": false,
    "delta_S_od": -0.00025610778784490584,
    "delta_S_link": 0.0,
    "delta_S_dev": 0.0032336170181077595,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": 0.0003697940481859341
  },
  {
    "phase": "sparse_stress",
    "scenario": "overall",
    "all_components_nondecreasing": false,
    "delta_S_od": -0.005778859307518591,
    "delta_S_link": 0.0,
    "delta_S_dev": -0.029018371309569857,
    "delta_S_attr": 0.0,
    "delta_synthetic_S_ODME": -0.00695324238481887
  }
]

诊断显示：重尾S_dev小升但S_od略降，稀疏阶段S_od和S_dev均降。即使S_link/S_attr不受损，也未实现联合稳健改善，因此保留v11。不得把校准阶段的均值上升或计数CV改善当作真实路径提升。

## 复现与交付

真实原始文件约181 MB，保存在work中，不上传GitHub。sources.json、pinned_sources.json和官方checksum验证记录保证来源可复现；聚合直方图、县级OD表、审计与失败模型全部保留。

--stage external-od-data下载并校验/清洗真实来源；--stage external-od运行完整本轮；--stage external-od-infer只复现冻结回退策略；--stage external-od-probe只复现诊断。默认v9入口及原始正式提交保留，未提交Kaggle。所有实际隐藏路径分数仍未知。
