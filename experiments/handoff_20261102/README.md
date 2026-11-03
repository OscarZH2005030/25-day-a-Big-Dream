# 0.81135 冻结提交交接与 Task1–Task3 误差诊断

本目录保存2026-11-02交接材料、可运行复核脚本和聚合证据。**不替换仓库已有模型或其他研究目录，不宣称Task3或Task4已获得官方提升，不触发Kaggle提交。**

## 已有官方对照

|版本|Kaggle提交编号|公开榜总分|
|---|---:|---:|
|冻结Task4基线|57900964|**0.81135**|
|Task4 v20|57900990|0.78677|

两份非Task4行逐字节一致；候选总分下降0.02458，折算Task4综合分约下降0.12290。没有官方分项或私榜分数。Task1的0.94282863197和Task2的0.827225403是此历史提交使用方案的本地指标；不覆盖仓库其他更新后的候选或口径。

## 新窗口先读

- [完整交接背景](NEW_CHAT_SUMMARY.md)
- [今日发现与公式解释](TODAY_FINDINGS.md)
- [移机说明](README_FIRST.md)
- [官方回执和完整提交验收](results/submissions/)
- [697070个公开历史目标的同权重诊断](results/task1_physics_sample/summary.json)
- [移机环境和推理检查](verification/)

Task1单点RMSE不直接评价误差在时间上的排列。Task3由同一预测计算 `k=q/max(v,1)`、`N=L*k`，再检查五分钟存量变化。相同边界证据下，守恒残差变化为 `e_N(t+5)-e_N(t)`，前后反号可使误差叠加。但样本不支持整体持续负相关交替；50.07%指同点速度/流量误差反号，46.13%才指相邻两端都含预测时的存量误差反号。

105个密度误差超过50%的点贡献20.40%密度平方误差，却只贡献1.07%密度绝对误差；相关转移贡献约1.38%存量差分绝对误差。不能把SSE贡献当官方LWR扣分贡献。建议研究联合状态/存量/真实变化监督，**该新训练方案尚未实现或训练，不能称为提分结果**。

## 完整移机文件

完整包的指纹和分卷顺序见 [FULL_LOCAL_PACKAGE.json](FULL_LOCAL_PACKAGE.json) 与 [release_assets.json](release_assets.json)：

- 原ZIP：`TFB_20261102.zip`，16,015,015,925字节。
- SHA256：`0044b1b4f72d6c9e60d6aab3522f4f4295d4f4945f1024c9c6cae93ad222e209`。
- 解压内容24,442,236,663字节、39,888个payload文件，另有两份清单。
- 原包已通过全部ZIP成员CRC和5,540份锁定原始输入哈希；桌面复制SHA一致。
- 代码、正式权重、完整原始数据、两套提交CSV/ZIP、610份实际Task1日预测、Task2/R127完整依赖、Task4研究、今天180份诊断Parquet、便携CPython/ML依赖均在完整包。

GitHub每个Release附件必须小于2GiB，因此原ZIP已无损切为9份，每份不超过2,000,000,000字节。没有把16GB二进制包或原始观测写入Git历史。

**完整包含原始竞赛输入，拟保存为仅有仓库push权限的协作者可见的草稿Release**，tag为 `handoff-windows-20261102-081135`。本目录的清单只表示本地准备完成，不单独证明远端附件已上传；实际远端状态以Release页面/上传回执为准。不要未经核对把含原始数据的草稿改成公开发布。GitHub草稿只对有写权限的账户可见，新电脑需登录相应团队账户查看。

取齐该Release的9个 `.partNNN`、`handoff-parts.json`、`Restore_From_Parts.ps1`、`Extract_Handoff.ps1` 后，在同一文件夹运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Restore_From_Parts.ps1
```

脚本核对每卷SHA、合并并核对完整ZIP SHA；不会执行训练或上传。随后运行 `Extract_Handoff.ps1` 解压。完整包内工具以解压后根目录为ROOT，不是把本Git目录当数据/运行环境根目录。

原始数据仍遵从比赛规则，来源为官方Kaggle Data页，许可证README保留；组织者隐藏标签和私有边界通量从未取得或打包。凭据/账户配置不在包内。完整包是既有0.81135提交的冻结快照，不能用其中历史目录覆盖本仓库较新的研究结果。

## 复核范围

便携环境已在原机的新路径检查：Task1单日5884行与实际提交缓存差异0；Task2烟测34800行通过；Task4基线20个panel/split对照最大差约4.55e-13；60个历史日诊断逐日SSE与发布记录最大相对差约1.37e-7。新电脑GPU/驱动需自行运行环境检查，本地诊断不等于官方Task3分数。

官方来源：[评分规则](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md)、[本地评分限制](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/RUN_LOCAL.md)。[下载的官方源码快照及来源哈希](docs/official_public_snapshot/)仅用于审计，不是私有Kaggle评估器。
