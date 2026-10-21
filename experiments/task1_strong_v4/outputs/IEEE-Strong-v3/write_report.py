"""Report only against the current v2 model; retain all previous artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from residual_core import *
from verify_delivery import main as verify_delivery

verify_delivery()

result=json.loads((ROOT/'evaluation-summary.json').read_text())
frozen=json.loads((ROOT/'frozen-routes.json').read_text())
monitor=json.loads((ROOT/'monitor-summary.json').read_text())
by=pd.read_csv(ROOT/'comparison-by-panel.csv')
routes=pd.DataFrame(frozen['routes'])
before=result['baseline'];after=result['new']
if (ROOT/'candidate-submission/submission-validation.json').exists():
    validation=json.loads((ROOT/'candidate-submission/submission-validation.json').read_text())
    submission=ROOT/'candidate-submission/submission.csv'
else:
    validation=json.loads((BASE/'candidate-submission/submission-validation.json').read_text())
    submission=BASE/'candidate-submission/submission.csv'
rows=[]
for panel in PANELS:
    z=by[by.panel==panel].iloc[0]
    rows.append(dict(panel=panel,baseline_score=float(z.S_state_baseline),new_score=float(z.S_state_new),
        score_delta=float(z.score_delta),baseline_macro_flow=float(z.rmse_flow_per_lane_baseline),
        new_macro_flow=float(z.rmse_flow_per_lane_new)))
details=pd.DataFrame(rows);details.to_csv(ROOT/'priority-directions.csv',index=False)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
for ax,key,title in [(axes[0],'score','Same 55-day local holdout score'),(axes[1],'flow_rmse','Pooled flow error (veh/hour/lane)')]:
    values=[before[key],after[key]];bars=ax.bar(['Current v2','Flow correction v3'],values,color=['#4878a8','#16806a'])
    margin=max(abs(values[1]-values[0])*.5,.0001 if key=='score' else .05)
    ax.set_ylim(min(values)-margin,max(values)+margin*2);ax.set_title(title)
    for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+margin*.15,f'{v:.6f}' if key=='score' else f'{v:.4f}',ha='center')
    ax.grid(axis='y',alpha=.15)
fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
text=f'''从当前新模型继续优化：流量误差修正实验

唯一比较基线：IEEE-Strong-v2。原 Transformer 当前不可用，本报告不比较它。
相同 55 天、6,636,705 个 Task 1 缺失目标；这不是分类准确率或 Kaggle 隐藏测试分数。

                    当前 v2             本轮 v3
S_state             {before['score']:.9f}        {after['score']:.9f}
每车道流量 RMSE     {before['flow_rmse']:.6f}           {after['flow_rmse']:.6f} 辆/小时/车道
速度 RMSE           {before['speed_rmse']:.6f}            {after['speed_rmse']:.6f} km/h
评分变化：{result['score_delta']:+.9f}；流量 RMSE 变化：{result['flow_rmse_delta']:+.6f}。
速度输出保持当前模型原值；没有重新训练或修改速度分支。
{result['adopted']}/3 个重点方向通过内部验证，{result['improved_directions']}/10 方向在本次留出评估评分提高。

三个重点方向
{details.to_string(index=False,float_format=lambda x:f'{x:.9f}')}
本表的流量列是三种遮挡 RMSE 的等权均值，整体流量为按全部目标数合并后的 RMSE。

方法
冻结当前 v2 模型，让附加网络只学习尚未预测准确的流量误差。
输入为 134 个局部/历史/观测特征，增加当前主干预测的速度与最终流量，共 136 个特征。
附加网络包含 32 维站点嵌入、512 宽度、4 个残差块；零初始化输出头。
浮点 32 位的当前流量预测单独保留，网络初始输出与当前模型完全一致，避免特征半精度量化改变初始结果。
优化目标为流量平方误差加 0.05 倍修正幅度惩罚，只输出非负流量。
最大 500 次更新，在第 100、300、500 次检查内部验证，EMA 0.95。
混合比例为 0、0.1、0.25、0.5、0.75、1；零代表完全保留当前模型。
要求完整内部验证、前 11 天、后 11 天的汇总与遮挡等权误差均不恶化，且汇总误差至少改善 0.01，才采用修正。

最终内部选择
{routes[['panel','alpha','step','pooled_flow','macro_flow','batch','parameters']].to_string(index=False,float_format=lambda x:f'{x:.6f}')}

数据与边界
监督训练期为 2030-06-29 至 12-13；内部验证为 2030-12-14 至 2031-01-04。
历史统计仅来自监督训练期；训练样本使用排除自己当天的历史统计。
残差训练使用冻结主模型在训练期的预测，属于主模型的样本内误差；这可能带来残差分布差异。
内部验证期不参与模型参数的梯度训练，用于决定检查点和混合比例。
所有三方向选择写入 frozen-routes.json 后才评估本轮留出数据。
此前已查看过该留出集并据此提出优化目标，因此结果依然属于探索性本地开发；单种子，不能保证隐藏测试提升。

资源与保留
准备冻结预测、训练和内部验证阶段耗时 {monitor['seconds']/60:.2f} 分钟。
显存采样峰值 {monitor['peak_vram_mib']/1024:.2f} GiB，GPU 功耗峰值 {monitor['peak_power_w']:.2f} W，GPU 温度峰值 {monitor['peak_temperature_c']:.1f} °C。
批量来自真实前向/反向显存探测；没有无效填充或额外压力运算。功耗不是整机插座功耗。
当前 v2 权重和提交文件已通过 SHA256 未改变核对。

文件
冻结的修正路由：frozen-routes.json；它依赖相邻 IEEE-Strong-v2 模型、历史统计和完整预测。
完整候选提交：{submission}
结构校验：{validation['status']}，共 {validation['rows']:,} 行，SHA256 {validation['sha256']}。
候选只更改采用修正的方向的流量；速度、其他方向和 Task 2/4 沿用当前 v2。
未上传 Kaggle。
'''
(ROOT/'优化结果与说明.txt').write_text(text,encoding='utf-8')
save(ROOT/'delivery.json',dict(status='completed',baseline='IEEE-Strong-v2',report='优化结果与说明.txt',metrics='evaluation-summary.json',
    routes='frozen-routes.json',submission=str(submission.resolve()),validation=validation['status'],previous_best_preserved=True,
    score=after['score'],flow_rmse=after['flow_rmse']))
print(text,flush=True)
