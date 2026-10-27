# v12：稀疏先验调整，失败方法保留

沿用固定480场景。新凸目标为计数平方误差 + 0.05/2路径调整平方 + eta路径调整L1范数，之后再次投影到原始计数。三个阈值、四种混合比例，连同v11和官方共14候选。check_solver.py与独立SLSQP绝对值辅助变量QP核对通过，solver_validation.json保留数值误差。

选择同时约束calibration总体及局部漏计、全局计数偏高两误差族的四项及综合分不降，另要求真实计数划分平均误差不增。本轮所有方向都保留v11；人工目标S_dev的改善无法同时通过其他条件，不部署失败方法。

所有候选完整结果：

        phase           candidate     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
  calibration  elastic0.25_mix0.1 0.506637 0.997381 0.653915 0.905704          0.711275
  calibration elastic0.25_mix0.25 0.505742 0.997398 0.653403 0.905518          0.710772
  calibration  elastic0.25_mix0.5 0.504098 0.997428 0.652513 0.905196          0.709857
  calibration    elastic0.25_mix1 0.501164 0.997486 0.651147 0.904498          0.708242
  calibration     elastic1_mix0.1 0.507500 0.997381 0.653532 0.905753          0.711613
  calibration    elastic1_mix0.25 0.507784 0.997398 0.652422 0.905630          0.711560
  calibration     elastic1_mix0.5 0.507770 0.997427 0.650497 0.905397          0.711237
  calibration       elastic1_mix1 0.505893 0.997486 0.643664 0.904780          0.709290
  calibration     elastic4_mix0.1 0.509169 0.997381 0.652260 0.905885          0.712193
  calibration    elastic4_mix0.25 0.511776 0.997398 0.649017 0.905940          0.712893
  calibration     elastic4_mix0.5 0.514852 0.997427 0.643416 0.905938          0.713443
  calibration       elastic4_mix1 0.513308 0.997485 0.625340 0.905611          0.710003
  calibration            official 0.498240 0.997363 0.654881 0.904470          0.707452
  calibration                 v11 0.507200 0.997369 0.654321 0.905824          0.711604
   evaluation  elastic0.25_mix0.1 0.495919 0.997232 0.661605 0.900681          0.706814
   evaluation elastic0.25_mix0.25 0.495211 0.997252 0.661148 0.900512          0.706407
   evaluation  elastic0.25_mix0.5 0.493982 0.997285 0.660527 0.900219          0.705725
   evaluation    elastic0.25_mix1 0.491389 0.997351 0.659343 0.899578          0.704301
   evaluation     elastic1_mix0.1 0.496706 0.997232 0.661664 0.900720          0.707183
   evaluation    elastic1_mix0.25 0.497082 0.997252 0.661338 0.900602          0.707291
   evaluation     elastic1_mix0.5 0.497435 0.997285 0.660365 0.900379          0.707279
   evaluation       elastic1_mix1 0.496444 0.997351 0.657613 0.899815          0.706352
   evaluation     elastic4_mix0.1 0.498359 0.997232 0.661318 0.900825          0.707891
   evaluation    elastic4_mix0.25 0.500972 0.997252 0.659991 0.900851          0.708876
   evaluation     elastic4_mix0.5 0.504624 0.997285 0.656512 0.900829          0.710003
   evaluation       elastic4_mix1 0.505336 0.997351 0.647938 0.900478          0.709001
   evaluation            official 0.488663 0.997212 0.662204 0.899558          0.703466
   evaluation                 v11 0.496551 0.997219 0.661939 0.900791          0.707162
sparse_stress  elastic0.25_mix0.1 0.443032 0.996643 0.648973 0.887247          0.678958
sparse_stress elastic0.25_mix0.25 0.441577 0.996724 0.648351 0.886978          0.678190
sparse_stress  elastic0.25_mix0.5 0.439625 0.996858 0.647215 0.886521          0.677106
sparse_stress    elastic0.25_mix1 0.435568 0.997126 0.644200 0.885571          0.674753
sparse_stress     elastic1_mix0.1 0.443111 0.996643 0.648990 0.887237          0.678995
sparse_stress    elastic1_mix0.25 0.441766 0.996724 0.648360 0.886952          0.678273
sparse_stress     elastic1_mix0.5 0.439970 0.996858 0.647184 0.886467          0.677249
sparse_stress       elastic1_mix1 0.436149 0.997126 0.643871 0.885457          0.674948
sparse_stress     elastic4_mix0.1 0.443404 0.996643 0.649254 0.887220          0.679164
sparse_stress    elastic4_mix0.25 0.442458 0.996724 0.648938 0.886906          0.678664
sparse_stress     elastic4_mix0.5 0.441218 0.996858 0.648033 0.886368          0.677923
sparse_stress       elastic4_mix1 0.438259 0.997126 0.645061 0.885236          0.676043
sparse_stress            official 0.435081 0.996579 0.646841 0.885471          0.674778
sparse_stress                 v11 0.443977 0.996590 0.649332 0.887418          0.679450

冻结策略：
{
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
}

数据和场景之前已查看；不是独立盲测，没有实际隐藏路径分数。
