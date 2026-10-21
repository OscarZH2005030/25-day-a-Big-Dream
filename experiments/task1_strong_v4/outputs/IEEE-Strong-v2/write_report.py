"""Generate the comparison report from completed experiment artifacts."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimize import ROOT,V1,save

r=json.loads((ROOT/'evaluation-summary.json').read_text())
monitor=json.loads((ROOT/'monitor-summary.json').read_text())
by=pd.read_csv(ROOT/'comparison-by-panel.csv')
internal=pd.read_csv(ROOT/'internal-selection.csv')
frozen=json.loads((ROOT/'frozen-routes.json').read_text())
local=json.loads((ROOT/'final-model.json').read_text())['local_flow_routes']
local_monitor=json.loads((ROOT/'local-flow/monitor-summary.json').read_text())
peak_vram=max(monitor['peak_vram_mib'],local_monitor['peak_vram_mib'])
peak_power=max(monitor['peak_power_w'],local_monitor['peak_power_w'])
peak_temperature=max(monitor['peak_temperature_c'],local_monitor['peak_temperature_c'])
bootstrap=json.loads((ROOT/'paired-bootstrap.json').read_text())
preserved=json.loads((ROOT/'preservation-check.json').read_text())
priority=['D12_I5_N','D7_I405_N','D7_I210_E']
compact=by[['panel','S_state','S_state_previous','S_state_new','rmse_flow_per_lane_previous','rmse_flow_per_lane_new','delta_vs_previous','delta_vs_transformer']]
compact=compact.set_index('panel').loc[priority+[p for p in compact.panel if p not in priority]].reset_index()
compact.to_csv(ROOT/'direction-summary.csv',index=False)
table=pd.DataFrame([dict(model=name,**r[key]) for name,key in [('原 Transformer','transformer'),('上一版最佳模型','previous'),('本轮冻结模型','new')]])
table.to_csv(ROOT/'model-summary.csv',index=False)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(16,5.4),layout='constrained')
labels=['Team Transformer','Previous best','Flow-focused v2']
for ax,column,title,unit in [(axes[0],'score','Matched local holdout score','S_state (higher is better)'),
                             (axes[1],'flow_rmse','Pooled flow error','veh/hour/lane (lower is better)')]:
    values=[r[key][column] for key in ['transformer','previous','new']]
    bars=ax.bar(labels,values,color=['#999999','#4878a8','#16806a'])
    lo=min(values);hi=max(values);margin=max((hi-lo)*.35,.001 if column=='score' else .5)
    ax.set_ylim(lo-margin,hi+margin*1.7)
    for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+margin*.15,f'{value:.6f}' if column=='score' else f'{value:.3f}',ha='center',fontsize=9)
    ax.set(title=title,ylabel=unit);ax.tick_params(axis='x',rotation=22);ax.grid(axis='y',alpha=.15)
z=compact.iloc[::-1]
axes[2].barh(z.panel,z.delta_vs_previous,color=['#a35348' if v<0 else '#16806a' for v in z.delta_vs_previous])
axes[2].axvline(0,color='#888',linewidth=.8)
axes[2].set(title='Per-corridor score change vs previous',xlabel='S_state delta');axes[2].grid(axis='x',alpha=.15)
fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
v=r['new'];p=r['previous'];tf=r['transformer']
text=f'''IEEE 第二轮优化结果

本轮冻结模型与上一版使用相同的 55 天、6,636,705 个目标进行比较。
这是 Task 1 的本地开发结果，不是准确率、Kaggle 隐藏测试分数或名次。

{table.to_string(index=False,float_format=lambda x:f'{x:.9f}')}

相对上一版 S_state 变化：{v['score']-p['score']:+.9f}
流量 RMSE 变化：{v['flow_rmse']-p['flow_rmse']:+.6f} 辆/小时/车道。
速度 RMSE 变化：{v['speed_rmse']-p['speed_rmse']:+.6f} km/h。
流量相对原 Transformer：{v['flow_rmse']-tf['flow_rmse']:+.6f}。
{r['directions_improved_vs_previous']}/10 方向高于上一版，{r['directions_improved_vs_transformer']}/10 高于原 Transformer。

三个重点方向及其余方向
{compact.to_string(index=False,float_format=lambda x:f'{x:.9f}')}
该表的分方向流量列为三类遮挡 RMSE 均值；整体表为按全部目标数汇总的 RMSE。
priority-directions.csv 另列三个重点方向按目标数汇总的速度/流量 RMSE 与评分。

这次改变
继续训练上一版各方向模型，保持 64 特征、完整一天 288 步、整条走廊和 11,388,974 参数。
流量损失权重提高，并增加流量平方误差项，让大误差对训练产生更多影响。
每方向最多 3 轮 × 20 次更新，基础学习率 3e-5，时间注意力学习率 6e-5，EMA 0.9。
沿用各方向已经验证的真实显存批量；FP16 和激活检查点用于有效上下文与反向传播。
第 5 方向原批量 8 在第二次更新时显存分配不足，该次尝试已归档且未进入最终模型。
随后限制单次批量×节点数不超过 704，降低单次批量并增加梯度累积，保持有效批量；前四方向结果保留。
重启的方向从上一版权重重新开始，训练损失、学习率、更新次数和验证规则不变。
每一轮在内部验证比较速度和流量分别混合上一版与新输出的 0、0.25、0.5、0.75、1 五种比例。
每个通道同时要求汇总 RMSE 和三类遮挡等权 RMSE 不恶化，零比例代表沿用上一版。
单方向最终选择一个权重版本及两个混合比例，以内部流量汇总 RMSE 优先，且综合分不能下降。
至少降低 0.02 辆/小时/车道且综合分增加 1e-5 才接受新检查点。
本轮 {r['adapted_routes']}/10 方向采用新检查点，其余保留上一版。

补充局部流量模型
对三个重点方向另外训练每套约 429 万参数的局部流量残差网络，宽 512、4 个残差块。
输入为原 64 特征、八类观测/初始值/质量特征在前后 5、15、30、60 分钟的数值，另加训练期同星期/时刻的流量均值、标准差和支持数量。
还加入根据当天可见观测校正幅度的历史轮廓、局部可见残差校正轮廓、局部观测比例，合计 134 通道。
这些校正只读取输入中可见的流量，不读取当前日隐藏目标。
历史只统计监督训练的 168 天，训练样本排除自己当天的贡献；内部验证和之后日期不使用自身或验证/留出标签建立历史。
另有 32 维站点嵌入。只监督训练期的缺失目标，只输出流量，不改变速度。
每方向最多 300 次更新，在第 100、200、300 次更新比较，候选混合比例为 0、0.05、0.1、0.2、0.3、0.5、0.75、1。
只有内部验证前 11 天、后 11 天、完整 22 天的汇总与遮挡等权误差全部不恶化，才允许采用。
本轮 {r['local_flow_routes']}/3 个重点方向采用该补充分支。
{pd.DataFrame(local)[['panel','alpha','step','pooled_flow','macro_flow']].to_string(index=False,float_format=lambda x:f'{x:.6f}')}
该局部模型与全天图模型互补；没有用本地留出标签训练或选择混合比例。

选择与评估边界
监督训练仍为 2030-06-29 至 12-13，内部验证为 2030-12-14 至 2031-01-04。
本地留出仍为 2031-01-05 至 02-28，没有将其标签加入训练。
全部方向的权重及比例在 frozen-routes.json 中冻结后，才启动本轮留出评估。
内部验证不恶化，不能保证未来或留出数据上逐方向也不恶化；应以完整结果表判断。
之前的本地留出结果已被查看并用于决定此次优化目标，因此依然是探索性开发。
只有单种子，不能据此保证泛化或竞赛名次。
按同一天配对重采样的 95% 描述区间：评分差 {bootstrap['score_delta_95pct']}；流量 RMSE 差 {bootstrap['flow_rmse_delta_95pct']}。
该区间以已经选择的模型为条件，不能消除历史查看留出数据造成的选择偏差。

资源与保留
主模型训练阶段耗时 {monitor['seconds']/60:.2f} 分钟，补充模型阶段 {local_monitor['seconds']/60:.2f} 分钟，均含验证。
两阶段 GPU 显存采样峰值 {peak_vram/1024:.2f} GiB。
GPU 功耗峰值 {peak_power:.2f} W、温度峰值 {peak_temperature:.1f} °C；不是整机插座功耗。
上一版 {preserved['files']} 个模型、校准、拓扑、路由与提交文件通过 SHA256 未改变核对。
原结果仍在 ../IEEE-Strong-v1/，本轮全部结果独立存放。

复现入口
optimize.py：训练与内部选择；evaluate.py：冻结后的统一评估；build_candidate.py：原始比赛数据推理与格式校验。
final-model.json：完整路由，包含原模型路由、追加权重、局部流量模型和全部混合比例。
frozen-routes.json 是第一阶段冻结记录；local-flow/policy.json 是第二阶段预先记录的策略。
本轮依赖相邻 IEEE-Strong-v1 中的骨干代码、原模型、历史统计和特征构建器，不能只复制一个追加权重文件。
comparison-by-regime.csv、direction-summary.csv、paired-daily-errors.csv 提供细分结果。
新提交存放在 candidate-submission/submission.csv；Task 2/4 继续沿用已验证基线。
没有上传 Kaggle。
'''
candidate=ROOT/'candidate-submission/submission-validation.json'
if candidate.exists():
    validation=json.loads(candidate.read_text())
    text+=f"\n提交结构校验：{validation['status']}，{validation['rows']:,} 行，SHA256 {validation['sha256']}。\n"
(ROOT/'优化结果与说明.txt').write_text(text,encoding='utf-8')
save(ROOT/'delivery.json',dict(status='completed',metrics='evaluation-summary.json',report='优化结果与说明.txt',
    plot='comparison.png',routes='final-model.json',submission='candidate-submission/submission.csv',
    previous_best_preserved=True,previous_score=p['score'],new_score=v['score'],previous_flow_rmse=p['flow_rmse'],new_flow_rmse=v['flow_rmse']))
print(text)
