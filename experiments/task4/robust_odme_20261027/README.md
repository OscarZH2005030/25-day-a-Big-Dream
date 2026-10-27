# Task 4 ODME：可复现基准与稳健性实验（2026-10-27）

本目录保存 Task 4 的非负求解器、外部数据取得与审计、实验配置、模型、逐场景结果、预测及验收。**目前没有真实比赛的 S_od、S_dev、S_attr 或完整 S_ODME 成绩。** 所有标为 synthetic/artificial/人工的分数仅用于方法敏感性研究。

## 已知结果与限制

- 冻结官方目标基准：validation 发布计数 `S_link=0.9999394100`，private 发布计数 `S_link=0.9999232335`。private 数字不是隐藏私榜成绩。
- 最新探索候选：先按当前 split 的公开计数校正先验加载尺度，再做 KL 正则化非负求解。只更新有同方向五分钟实测流量支持的 D7 I10 E/W、I405 N/S，其余六方向沿用冻结基准。
- 新尺度匹配人工场景中，旧幅度候选 / scaled KL 的 S_dev 为 0.67423 / 0.92387，人工综合为 0.66940 / 0.88626。这依赖生成假设，不能称作真实提升，不能与早期不同场景的 0.95122 直接比较。
- 实际 train 四方向分组计数留出 NMAE：官方基准 0.030782、旧幅度候选 0.043517、scaled KL 0.037156。官方基准在这一检查中仍最好，所以未覆盖主预测。
- v6 自适应模型未通过 S_dev 改善条件；v7 在 160 个人工场景中有 133 个未达到幅度校准目标，**不可用于选模**。保留失败记录，未挑选性删除。
- 参数选择和实际独立路径评估不能混为一谈。已多轮查看同一公开 train 和外部日期；没有隐藏路径真值。最新候选尚未自动提交 Kaggle、合并或发布为获胜模型。

## 文件与目录

- `reproduce.py` / `reproduce.ps1`：可迁移统一入口。
- `requirements.txt`：数值环境；`requirements-data.txt`：XLSX 提取时使用的数据环境。
- `competition_input_manifest.json`：官方输入的路径、大小和 SHA256。**不包含原始竞赛数据。**
- `outputs/run_task4.py`：官方 λ=0.05 目标的等价对偶求解及基准比较。
- `outputs/scale_shape_v8/infer_balanced.py`：最新候选的独立推理，无需重跑人工实验。
- `outputs/task4_path_flow.csv`：冻结基准预测。
- `outputs/scale_shape_v8/task4_balanced_scale_candidate.csv`：最新探索候选。
- `outputs/{improvement_v2,external_v3,fivemin_v4,dev_v5,adaptive_v6,magnitude_v7,scale_shape_v8}`：历轮脚本、配置、结果及说明。冗余的逐方向/逐候选预测可由代码重建，未全部重复提交。
- `outputs/official_reference`：官方代码、配置与文档快照，提交 `f9e3fbc35c066419726987e470448f33d4c72fd9`，保留 MIT 许可证及原作者来源。

## 准备环境

Python 3.12；本机数值运行版本是 3.12.14。CPU 默认4线程，不使用 GPU。

```powershell
python -m venv .venv-model
.venv-model\Scripts\python.exe -m pip install -r requirements.txt
```

如需从头重建 Caltrans XLSX 的清洗数据，建议另建数据环境，保持提取环境与实验记录一致：

```powershell
python -m venv .venv-data
.venv-data\Scripts\python.exe -m pip install -r requirements-data.txt
```

已提交的清洗表及 PM 统计可直接支持推理和数值实验；仅运行最新推理无需下载外部原始数据或安装 openpyxl。

## 官方输入

从 [Kaggle 官方 Data 页](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/data) 按规则下载 `kaggle_public`。至少需要 `config/`、十方向的 `task4/` 和 `corridors/*/network/{path_set,path_link_incidence,links}.csv`。基准统一提交键验收还需 `submission_key.csv`。建议保持完整原始发布目录，不改动任何原始文件。

本目录不会上传原始竞赛数据、账号信息或隐藏标签。第一次 `verify` 会检查所需小文件哈希，发现版本不符即停止。

## 快速推理与验收

在本目录执行，路径可含空格：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\reproduce.ps1 -ReleaseRoot 'D:\data\kaggle_public' -Python '.venv-model\Scripts\python.exe' -Stage latest
```

跨平台等价入口：

```bash
python reproduce.py --release-root /data/kaggle_public --stage latest
python reproduce.py --release-root /data/kaggle_public --stage verify
```

`latest` 从冻结基准起构建四方向探索候选，随后核对所有已保存完整预测的模板键、区域、period token、非负/有限性和重复。每份完整预测应为70,708行，覆盖validation/private十方向。`departure_time` 是原样保留的period token，不转时间戳。

## 重现所有已完成实验

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\reproduce.ps1 -ReleaseRoot 'D:\data\kaggle_public' -Python '.venv-model\Scripts\python.exe' -DataPython '.venv-data\Scripts\python.exe' -Stage all
```

阶段：`baseline` 重建基准并验收；`external` 下载/提取外部资料；`all` 按历轮顺序运行，**包括已知失败的代理方法以保留可复现记录**，不表示推荐它们。该入口只写本包输出与 `work/`，不写官方输入。全量实验比最新推理更慢；具体旧运行时间见各 manifest，网络下载时间另计。

## 外部数据与许可

详见 [DATA_SOURCES.md](DATA_SOURCES.md)。公开数据的固定版本URL与SHA256位于 `outputs/fivemin_v4/sources.json`，Caltrans年度资料来源位于 `outputs/external_v3/sources.json`。原始外部大文件通过脚本下载，避免重复存放；数据来源、许可证与派生统计随本目录提供。

## 提交与其他任务

Task 1、2、4可以使用不同模型。Task 3直接评估Task 1同一份输出，没有单独提交。本目录只处理Task 4，未更改Task 1权重、Task 2代码或其他任务输出。需由使用者最后按官方提交键合并；本目录不自动提交Kaggle。
