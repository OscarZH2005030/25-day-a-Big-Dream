# 数据来源、许可与处理

1. **TrafficFlowBench 官方发布**：https://github.com/jacky850/trafficflowbench-public 。代码快照 f9e3fbc35c066419726987e470448f33d4c72fd9，MIT许可证保留在 outputs/official_reference/LICENSE。竞赛原始数据须自行从Kaggle按竞赛规则取得，本包不分发。
2. **Caltrans年度资料**：https://dot.ca.gov/programs/traffic-operations/census 。D7/D12的2024版匝道ADT表、峰小时表、AADT GIS。2024版表并不表示每行都有2024观测；缺失年份不补零。目标五个道路家族有1193条匝道位置记录。来源与哈希见 outputs/external_v3/sources.json；清洗统计保留在对应目录。
3. **ASU Trans+AI Lab cbi_plus**：https://github.com/asu-trans-ai-lab/cbi_plus/tree/68be4363108c64499327b5cf5152410901aec393/benchmarks 。I10及I405的PeMS紧凑数据，61站×31天×288点=544608条，2018年3月，匿名可下载。流量按上游契约为vph/lane。紧凑数据不保留pct_observed；车道标签不保证可靠；速度单位说明存在冲突，所以本实验只用流量相对波动。保留 CBI_LICENSE.txt；软件为MIT，数据按上游公共机构资料声明及Caltrans条款使用。
4. **STGCN原作者PeMSD7**：https://github.com/VeritasYin/STGCN_IJCAI-18/tree/4ab75ab69af354d6a0ab86aefa082d389b187127/dataset 。2012年5–6月工作日块，五分钟速度，1026站；矩阵本身没有逐行时间戳。带道路号/方向/经纬度，不把速度反推后称为实测流量。保留 STGCN_LICENSE.txt，BSD-2-Clause。

Caltrans条款：https://dot.ca.gov/conditions-of-use 。网站一般信息视为public domain，但个别材料有例外；不把软件许可证误称为所有数据的统一授权。保留上游原有版权、引用与来源。

已排除的来源：USDOT AMS Pasadena表字段名含5-minute，但抽查时间戳间隔为一小时；SMU公开目录的CSV下载返回401；Seattle I5/I405不在加州。D12 I5/I405仍无已验证匿名取得的五分钟流量，因此没有把D7 I5冒充比赛D12 I5。

所有人工f*只用于方法实验，不是竞赛隐藏标签。发布的预测都是本模型计算结果，不含主办方保管真值。
