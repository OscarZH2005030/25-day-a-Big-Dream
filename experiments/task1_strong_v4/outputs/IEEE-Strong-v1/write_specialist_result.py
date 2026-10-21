"""Report all routed experts against the exact archived local targets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
route_manifest=json.loads((root/'final-model.json').read_text())
folders=[root/Path(v['checkpoint']).parent for v in route_manifest['routes']]
out=root/'specialists'
table=pd.concat([pd.read_csv(p/'calibrated-holdout-by-regime.csv') for p in folders],ignore_index=True)
raw_table=pd.concat([pd.read_csv(p/'holdout-by-regime.csv') for p in folders],ignore_index=True)
reference=pd.read_csv(root/'references/transformer-all-metrics.csv')
reference=reference[reference.regime.isin(['R1','R2','R3'])]
comparison=table.merge(reference[['panel','regime','n','S_state']],on=['panel','regime'],
    suffixes=('_new','_transformer'),validate='one_to_one')
assert len(comparison)==30 and (comparison.n_new==comparison.n_transformer).all()
assert int(comparison.n_new.sum())==6636705
comparison['delta']=comparison.S_state_new-comparison.S_state_transformer
comparison.to_csv(out/'comparison-with-transformer.csv',index=False)
table.to_csv(out/'calibrated-holdout-by-regime.csv',index=False)
raw_table.to_csv(out/'raw-holdout-by-regime.csv',index=False)
def score(frame):return float(frame.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean())
value=score(table);raw=score(raw_table)
tf=float(reference.groupby('panel').S_state.mean().mean());combo=.9369103120317692
def pooled(frame,column):return float(np.sqrt(np.dot(frame.n,frame[column]**2)/frame.n.sum()))
speed_rmse=pooled(table,'rmse_speed_kmh');flow_rmse=pooled(table,'rmse_flow_per_lane')
tf_speed_rmse=pooled(reference,'rmse_speed_kmh');tf_flow_rmse=pooled(reference,'rmse_flow_per_lane')
global_folder=root/route_manifest['global_source']
global_result=json.loads((global_folder/'calibrated-result.json').read_text())
details=[];seconds=0.;monitors=[]
for folder in folders:
    r=json.loads((folder/'training-result.json').read_text())
    initial=json.loads((folder/'initial-development-metrics.json').read_text())
    original=json.loads((folder/'original-development-metrics.json').read_text())
    graph_choice=json.loads((folder/'graph-selection.json').read_text())
    assert r['best_development_score']>=initial['score']-1e-6
    monitor=json.loads((folder/'monitor-summary.json').read_text());monitors.append(monitor)
    seconds+=r['seconds']
    details.append(dict(panel=folder.name,best_epoch=r['best_epoch'],
        original_internal=original['score'],initial_internal=initial['score'],best_internal=r['best_development_score'],
        proximity=graph_choice['use_proximity'],
        path_relation=graph_choice['use_path'],
        micro_batch=r['micro_batch'],accumulation=r['gradient_accumulation'],parameters=r['parameters']))
details=pd.DataFrame(details);details.to_csv(out/'expert-selection-details.csv',index=False)
by_panel=comparison.groupby('panel').agg(new=('S_state_new','mean'),transformer=('S_state_transformer','mean')).reset_index()
by_panel['delta']=by_panel.new-by_panel.transformer
by_panel.to_csv(out/'comparison-by-panel.csv',index=False)
result=dict(status='completed',score=value,raw_score=raw,transformer_score=tf,existing_hybrid_score=combo,
    global_model_score=global_result['calibrated_holdout_score'],global_source=global_folder.name,
    delta_vs_transformer=value-tf,delta_vs_existing_hybrid=value-combo,
    directions_improved=int((by_panel.delta>0).sum()),
    regressed_directions=by_panel.loc[by_panel.delta<0,'panel'].tolist(),
    pooled_speed_rmse_kmh=speed_rmse,pooled_flow_rmse_per_lane=flow_rmse,
    transformer_pooled_speed_rmse_kmh=tf_speed_rmse,transformer_pooled_flow_rmse_per_lane=tf_flow_rmse,
    targets=int(comparison.n_new.sum()),routes=10,adapted_routes=int((details.best_epoch>0).sum()),
    parameters_per_expert=int(details.parameters.iloc[0]),
    internal_initial=float(details.original_internal.mean()),internal_best=float(details.best_internal.mean()),
    proximity_routes=int(details.proximity.sum()),
    path_routes=int(details.path_relation.sum()),
    specialist_training_minutes=seconds/60,
    peak_vram_mib=max(v['peak_vram_mib'] for v in monitors),
    peak_gpu_power_w=max(v['peak_power_w'] for v in monitors),
    peak_gpu_temperature_c=max(v['peak_temperature_c'] for v in monitors),
    selection='Internal development only. No checkpoint or route is chosen by local holdout score.',
    limitation='Exploratory development on a historically public local holdout; no new hidden-test score or statistical-stability claim.')
(out/'summary.json').write_text(json.dumps(result,indent=2))
outcome='超过' if value>tf else '未超过'
outcome_combo='超过' if value>combo else '未超过'
text=f'''IEEE 新模型最终实验结果

结论：{outcome}原 Transformer；{outcome_combo}原 Transformer + LightGBM 孤立节点回退组合。

同一 55 天、6,636,705 个目标的 Task 1 S_state（越高越好）
团队 Transformer：{tf:.9f}
团队 Transformer + 孤立节点回退：{combo:.9f}
本轮所选全局模型：{global_result['calibrated_holdout_score']:.9f}
分道路模型，原始输出：{raw:.9f}
分道路模型，内部验证校准：{value:.9f}
相对 Transformer：{value-tf:+.9f}；相对原组合：{value-combo:+.9f}

全部目标汇总 RMSE（与等权宏平均 S_state 是不同统计量）
车速：新模型 {speed_rmse:.6f} km/h；原 Transformer {tf_speed_rmse:.6f} km/h。
每车道流量：新模型 {flow_rmse:.6f} 辆/小时/车道；原 Transformer {tf_flow_rmse:.6f}。
总体评分提高不代表所有指标都改善：{int((by_panel.delta>0).sum())}/10 个方向高于原 Transformer，
仍落后的方向为 {', '.join(by_panel.loc[by_panel.delta<0,'panel'])}。
本轮车速汇总 RMSE 下降，但流量汇总 RMSE 上升；请结合目标是宏平均评分还是单项误差来使用模型。

模型与新意
全局预训练后，每个方向单独细调；每套模型 {int(details.parameters.iloc[0]):,} 参数。
256 宽、10 个图时空块、双向实数对角状态空间、局部/全局修正头，另含两层全天时间注意力。
64 个输入特征，完整 288 步一天和整条走廊；推理时逐方向加载对应模型。
共享预训练用于迁移交通模式，分方向细调用于检验局部适应能否减少共享模型的限制。
另在内部验证比较原拓扑、仅补充孤立节点的距离衰减邻近关系、跨越无传感器路段的公开路径关系。
{int(details.proximity.sum())}/10 个方向采用距离关系，{int(details.path_relation.sum())}/10 个方向采用路径关系。
补充关系不是新增的直接 LWR 物理边；不修改主办方网络，输入特征仍按原物理图构建。
原 Transformer 为 width64、3 层、24 步窗口、空间 patch<=64 的逐方向模型。
图状态空间、注意力和暖扩展都有已有研究；不宣称本组合为全球首创。

特征
48 个基础通道包含插值、历史轮廓、缺口、观测、二跳上下游、匝道及可用性、道路属性与全局统计。
另有 16 个占有率、占有率可用性、相邻质量和插补标记通道。
占有率和插补标记是补回参考模型已有信息；不把旧特征作为新发明。
历史轮廓只拟合最初 28 天；额外传感器特征只来自公开 masked 层，不读取隐藏答案。
原始输入与缓存的 64 通道在十个方向逐位核对一致。

训练与选择
分道路阶段每方向最多两轮，每轮 30 次更新，共最多 60 次更新。
每轮验证；第零轮保留全局起点。{int((details.best_epoch>0).sum())}/10 个方向选择了细调后的权重。
所有模型权重按内部验证选，不根据本地留出结果选择道路或模型。
图关系候选也只按内部验证选择，至少提高 0.0001 才采用。
每方向仅两个残差系数，在内部验证拟合并冻结后应用于留出集。
显存批量按每条道路实际探测，微批量范围 {int(details.micro_batch.min())}–{int(details.micro_batch.max())}，梯度累积范围 {int(details.accumulation.min())}–{int(details.accumulation.max())} 次。
第六方向最初批量在实际累积反向传播时显存不足，失败发生在首次优化器更新前。
该尝试日志已归档，随后增加显存余量并按比例增加累积次数，保持相应有效批量；前五方向未重训。
此阶段训练及模型评估耗时 {seconds/60:.2f} 分钟，不含全局训练、特征准备及校准。

日期与比较边界
历史统计：2030-06-01 至 06-28；监督训练：06-29 至 12-13。
内部验证：2030-12-14 至 2031-01-04；本地留出：2031-01-05 至 02-28。
全部 30 个方向×遮挡 regime 的目标数与仓库 Transformer 记录一致。
按 regime、方向、道路家族等权评分，流量按每车道归一。
训练预算、特征、共享预训练和模型结构同时变化，不是等算力或等特征消融。
不能把整体增益单独归因为注意力、状态空间或任意一项特征。
第一阶段留出结果已查看后，才提出混合候选；分道路设置记录时混合候选还没有留出结果。
该本地留出此前也被团队多次查看，这是公开开发过程，不是未触碰外部测试。
只有每道路单种子；没有历史 Transformer 逐目标预测，不能作逐日配对显著性结论。
以上不是准确率、Kaggle 隐藏测试总分或名次。没有把本地留出标签加入这些比较模型的训练。

资源
本阶段 GPU 显存峰值 {result['peak_vram_mib']/1024:.2f} GiB /16 GiB。
GPU 功耗采样峰值 {result['peak_gpu_power_w']}W；这不是整机插座功耗。
计算用于完整上下文、模型与反向传播，没有额外压力运算或无效填充。

分方向结果
{by_panel.to_string(index=False,float_format=lambda v:f'{v:.9f}')}

入口
final-model.json：十个方向的权重与校准系数路由。
specialists/summary.json：完整汇总数值。
specialists/comparison-with-transformer.csv：30 组同目标对照。
specialists/expert-selection-details.csv：每方向最佳轮、内部验证与真实批量。
specialists/final-chart.png：内部验证改善与本地留出比较。
specialist_inference.py：生成候选提交；Task 2/4 复用已验证的基线预测。
旧 IEEE-Competition-Ready/submission.csv 保留；未向 Kaggle 上传。
'''
(root/'最终模型结果.txt').write_text(text,encoding='utf-8')
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
for folder in folders:
    h=pd.read_json(folder/'training-history.json').dropna(subset=['development_score'])
    base=json.loads((folder/'original-development-metrics.json').read_text())['score']
    axes[0].plot(h.epoch*30,h.development_score-base,'o-',label=folder.name)
axes[0].set(xlabel='Specialization updates',ylabel='Internal score change',title='Internal development, per corridor')
axes[0].legend(fontsize=7,ncol=2)
values=[tf,combo,global_result['calibrated_holdout_score'],raw,value]
labels=['Team TF','Team hybrid','New global','Experts raw','Experts calibrated']
bars=axes[1].bar(labels,values,color=['#888','#aaa','#96b7c9','#16756d','#315d9e'])
for bar,v in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,v+.0005,f'{v:.5f}',ha='center',fontsize=8)
axes[1].set(ylim=(min(values)-.01,min(1,max(values)+.006)),ylabel='Task 1 score',title='Matched 55-day local holdout')
axes[1].tick_params(axis='x',rotation=18)
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.savefig(out/'final-chart.png',dpi=150)
print(json.dumps(result,indent=2),flush=True)
