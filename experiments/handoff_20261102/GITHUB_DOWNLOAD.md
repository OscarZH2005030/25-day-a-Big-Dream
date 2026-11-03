# 在空白Windows电脑从GitHub恢复完整交接包

## 1. 登录有仓库写权限的团队账号

打开 [Oscar仓库的Releases](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases)，找到tag为 `handoff-windows-20261102-081135` 的草稿。

完整包含原始竞赛输入，保留为草稿；只有具备仓库push权限的账户可见。已登录普通GitHub账户仍可能看不到。请使用团队有写权限的账户；不需要把Token发给任何聊天。

交接源码和聚合报告在公开分支 `codex/handoff-20261102`，该分支不替换main中的较新模型或研究结果。

## 2. 下载并放进同一个文件夹

- `TFB_20261102.zip.part001` 到 `TFB_20261102.zip.part009`，共9份。
- `handoff-parts.json`。
- `Restore_From_Parts.ps1`。
- `Extract_Handoff.ps1`。
- 建议同时下载SHA256文件、总结和其余说明。

前8卷各2,000,000,000字节，第9卷15,015,925字节。GitHub自动提供的Source code ZIP只包含Git源码，不能代替这9份完整移机附件。

本流程会同时保留分卷约16.02GB、合并ZIP约16.02GB、解压内容约24.44GB，合计约56.47GB；建议至少65GB可用磁盘空间。后续全量训练缓存需要额外空间。合并脚本不自动删除分卷。

## 3. 合并并校验

在下载文件夹的PowerShell执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Restore_From_Parts.ps1
```

脚本先检查每卷大小/SHA256，按清单顺序合并，再校验完整ZIP：

```text
0044b1b4f72d6c9e60d6aab3522f4f4295d4f4945f1024c9c6cae93ad222e209
```

脚本拒绝覆盖不同内容的现有ZIP。若遇到缺卷或哈希错误，请重新下载对应卷，不要跳过检查。

## 4. 解压和检查

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\Extract_Handoff.ps1 -Destination D:\TrafficFlowBench
```

随后进入 `D:\TrafficFlowBench\TFB_20261102`，双击“检查环境.cmd”和“检查文件完整性.cmd”。包内自带可移动CPython和ML依赖，不需要先安装Python或Git。GPU仍需要适配的NVIDIA显卡驱动；需要VC++运行库时，包内installers提供已验证微软签名的安装器，由用户手动安装。

## 5. 开新聊天

粘贴解压目录中的NEW_CHAT_SUMMARY.md，并提供实际根目录。先阅读AGENTS.md；当前已知最好官方提交为0.81135，新多目标训练尚未实现。任何Kaggle提交仍需用户当次明确授权。

完整ZIP已经在原机通过全部成员CRC，锁定的5540份原始输入哈希匹配，并完成移机路径下Task1、Task2、Task4检查；新电脑请再次运行自带环境及文件完整性检查。
