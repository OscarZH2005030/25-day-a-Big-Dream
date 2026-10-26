# 十道路逐路选优模型

共同55日（2031-01-05至2031-02-28），6,636,705个官方目标，R1/R2/R3宏平均后对十路等权。

- 原GitHub Transformer＋孤立回退：**0.936910312**
- 逐道路选优：**0.940519388**（+0.003609075）
- 五路原GitHub组合，五路186特征适配LightGBM。验证期已被查看，本成绩是模型选择结果，不是独立盲测或Kaggle总分。

## 下载完整模型包

[完整模型、推理代码及中文说明](https://github.com/OscarZH2005030/25-day-a-Big-Dream/releases/tag/road-selection-common55-20261025)。模型权重放在Release ZIP里，仓库此目录保留源码、配置和审计，不能只clone源码就直接推理。

ZIP SHA256：`aa220eb17a275f6a82e8807cd6df52caec04dd1e9134b12df85bc419edcfe69f`。

解压后运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\模型目录\运行预测.ps1" -Date 2031-03-01 -PythonPath "C:\Python环境\Scripts\python.exe" -DataRoot "D:\数据目录\kaggle_public"
```

默认CPU，可加`-Device cuda`；完整中文说明见[中文说明.txt](中文说明.txt)，逐路成绩见[audit/逐路对比与选择.txt](audit/逐路对比与选择.txt)。

模型/历史文件全部在ZIP内以相对路径加载，不依赖旧训练目录；原始Kaggle数据和Python环境须另行准备。发布包在独立目录通过Windows PowerShell5.1十路实际推理验证，并与原验收预测逐项完全相同。
