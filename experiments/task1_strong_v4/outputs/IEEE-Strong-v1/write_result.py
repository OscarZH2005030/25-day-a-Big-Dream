"""Generate a factual report after training and pre-specified calibration."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=root/'training')
out=parser.parse_args().out
raw=json.loads((out/'training-result.json').read_text())
result=json.loads((out/'calibrated-result.json').read_text())
monitor=json.loads((out/'monitor-summary.json').read_text())
history=pd.read_json(out/'training-history.json')
comparison=pd.read_csv(out/'calibrated-comparison-with-transformer.csv')
panels=comparison.groupby('panel').agg(new=('S_state_new','mean'),transformer=('S_state_transformer','mean')).reset_index()
panels['delta']=panels.new-panels.transformer
panels.to_csv(out/'comparison-by-panel.csv',index=False)
state_score=result['calibrated_holdout_score'];ref=result['original_transformer_score']
outcome='超过仓库中单独 Transformer 的同范围留出分数' if state_score>ref else '未超过仓库中单独 Transformer 的同范围留出分数'
combo=result['existing_isolated_fallback_score']
outcome_combo='超过' if state_score>combo else '未超过'
alpha=result['calibration']['coefficients']
model_config=json.loads((out/'config.json').read_text())
architecture_description='缺口感知、多尺度图传播、双向实数对角状态空间、局部/全局双修正头。'
if model_config.get('variant')=='full_day_attention_ssm':
    architecture_description+='另含两层完整日时间注意力，以零输出投影从原模型平稳扩展。'
phase_note=('本阶段从前一阶段最佳内部验证权重继续细调。' if 'warm_start' in model_config else '本阶段为从头训练。')
chain=[];folder=out
while folder not in chain:
    chain.append(folder)
    source_config=json.loads((folder/'config.json').read_text())
    if 'warm_start' not in source_config:break
    folder=Path(source_config['warm_start']).parent
chain.reverse()
histories=[];total_seconds=0.;total_rounds=0;boundaries=[]
for folder in chain:
    phase=json.loads((folder/'training-result.json').read_text())
    frame=pd.read_json(folder/'training-history.json')
    if histories:frame=frame[frame.epoch>0].copy();boundaries.append(total_rounds)
    frame['epoch']+=total_rounds
    histories.append(frame);total_seconds+=phase['seconds'];total_rounds+=phase['trained_epochs']
plot_history=pd.concat(histories,ignore_index=True)
stage_rows=[]
for stage in ['training','refinement','hybrid']:
    stage_folder=root/stage
    if not (stage_folder/'calibrated-result.json').exists():continue
    stage_result=json.loads((stage_folder/'calibrated-result.json').read_text())
    stage_training=json.loads((stage_folder/'training-result.json').read_text())
    stage_rows.append(dict(stage=stage,rounds=stage_training['trained_epochs'],
        internal=stage_training['best_development_score'],raw=stage_result['raw_holdout_score'],
        calibrated=stage_result['calibrated_holdout_score'],
        selected=stage_folder.resolve()==out.resolve()))
stage_table=pd.DataFrame(stage_rows)
stage_table.to_csv(root/'candidate-summary.csv',index=False)
search_note=('额外候选结构和训练预算是在第一阶段本地留出结果已被查看后提出的；阶段内权重和校准只用内部验证。\n' if (root/'extended-development-policy.json').exists() else '')
text=f'''新图状态空间模型重训结果

结果：{outcome}。
与原 Transformer + LightGBM 孤立回退组合比较：{outcome_combo}。

模型：{architecture_description}
参数量：{raw['parameters']:,}；宽度 256；深度 10；{len(model_config['features'])} 个输入特征。
完整 288 步日序列、完整方向走廊；十个方向共享模型，道路身份嵌入。
微批量 {raw['micro_batch']}，梯度累积 {raw['gradient_accumulation']}。
训练轮数：{raw['trained_epochs']}；最佳内部验证轮：{raw['best_epoch']}。
阶段说明：{phase_note}
最佳内部验证分数：{raw['best_development_score']:.9f}。
本阶段训练及评估耗时：{raw['seconds']/60:.2f} 分钟。
所选模型训练链累计轮数：{total_rounds}；该链累计训练及模型评估耗时：{total_seconds/60:.2f} 分钟（不含特征准备和后续校准）。
这里的轮是固定更新预算，每轮 60 次优化器更新，不是完整遍历全部样本。

同一 55 天本地留出范围的 Task 1 S_state

团队单独图 Transformer：{ref:.9f}
团队 Transformer + LightGBM 孤立回退：{combo:.9f}
新图状态空间模型原始输出：{result['raw_holdout_score']:.9f}
新模型 + 预先规定的内部验证校准：{state_score:.9f}
校准版本相对 Transformer：{state_score-ref:+.9f}
校准版本相对原组合：{state_score-combo:+.9f}

本目录各已完成候选阶段（保留未胜出的结果；selected 只依据内部验证）
{stage_table.to_string(index=False,float_format=lambda v:f'{v:.9f}')}

速度修正系数：{alpha[0]:.7f}；每车道流量修正系数：{alpha[1]:.7f}。
系数只在内部验证 22 天选择，范围事先固定为 [0,2]。
内部验证原始分数：{result['calibration']['raw_development_score']:.9f}。
内部验证校准分数：{result['calibration']['calibrated_development_score']:.9f}。
系数冻结后才读取本轮保存的留出预测；没有根据留出结果在原始/校准之间择优。
原始输出与校准结果均已报告。校准不是额外的隐藏标签训练。

公平比较边界

历史统计：2030-06-01 至 06-28；训练监督：06-29 至 12-13。
内部验证：2030-12-14 至 2031-01-04。
本地留出：2031-01-05 至 02-28，共 55 天。
全部 30 个方向×regime 的目标数量与团队 Transformer 记录逐组一致。
总目标数：{int(comparison.n_new.sum()):,}。
按 regime、方向、道路家族等权聚合，流量按每车道评分。
新模型共享十方向信息，训练预算不同，不声称是等算力或等架构消融。
这是本地 Task 1 分数，不是准确率、Kaggle 排名或四任务总分。
仓库已说明该公开本地留出过去被多次查看，本轮不把它称为全团队未触碰测试。
{search_note}各阶段公开留出结果属于同一个开发过程，不能当作多个独立的外部验证。
没有原模型逐日预测，无法给出逐日配对显著性结论。
只有单随机种子，不把一次结果表述为统计稳定提升。

资源记录

显存峰值：{monitor['peak_vram_mib']/1024:.2f} GiB /16 GiB。
最高 GPU 温度：{monitor['peak_temperature_c']}°C；GPU 功耗采样峰值 {monitor['peak_power_w']}W。
这是显卡遥测功耗，不是笔记本整机的插座功耗。
资源用于模型、完整上下文和反向传播；未运行额外压力测试或活动伴随计算。

分方向结果

{panels.to_string(index=False,float_format=lambda v:f'{v:.9f}')}

复核文件

training/best-model.pt：按内部验证选择的 EMA 权重。
training/residual-calibration.json：推理使用的两个系数。
training/calibrated-comparison-with-transformer.csv：逐方向、逐 regime 比较与目标数。
training/training-history.json：全部轮次损失与内部验证。
training/accuracy-chart.png：学习曲线和留出对照。
training/*-holdout.npz：未校准的原始留出网格预测，单位为速度/25、每车道流量/600。
校准公式为 max(0,base+alpha*(raw_prediction-base))，base 来自输入前两个通道乘 [4,2000/600]。
校准版本推理需要权重与 residual-calibration.json，不能只使用权重宣称复现校准成绩。

旧 IEEE-Competition-Ready 提交文件没有被本轮候选覆盖。
当前产物是可审查的训练与留出结果，尚未获得主办方隐藏测试评分。
'''
text=text.replace('training/',out.name+'/')
(root/'重训结果.txt').write_text(text,encoding='utf-8')
(out/'重训结果.txt').write_text(text,encoding='utf-8')
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
axes[0].plot(plot_history.epoch,plot_history.training_loss,'o-',color='#16756d')
axes[0].set(xlabel='Training round',ylabel='Reconstruction objective',title='Training')
dev=plot_history.dropna(subset=['development_score'])
axes[1].plot(dev.epoch,dev.development_score,'o-',color='#315d9e',label='EMA candidate')
axes[1].axhline(float(dev.initial_score.iloc[0]),color='#888',ls='--',label='Initial reconstruction')
axes[1].set(xlabel='Training round',ylabel='Task 1 score',title='Internal development only');axes[1].legend(fontsize=8)
labels=['Team TF','Team hybrid','New raw','New calibrated']
values=[ref,combo,result['raw_holdout_score'],state_score]
bars=axes[2].bar(labels,values,color=['#888','#aaa','#16756d','#315d9e'])
for bar,value in zip(bars,values):axes[2].text(bar.get_x()+bar.get_width()/2,value+.0005,f'{value:.5f}',ha='center',fontsize=8)
axes[2].set(ylim=(max(0,min(values)-.015),min(1,max(values)+.006)),ylabel='Task 1 score',title='Matched 55-day local holdout')
axes[2].tick_params(axis='x',rotation=15)
for ax in axes:ax.grid(axis='y',alpha=.15)
for boundary in boundaries:
    for ax in axes[:2]:ax.axvline(boundary,color='#bbb',ls=':',lw=.8)
fig.savefig(out/'accuracy-chart.png',dpi=150)
print('Final report and learning chart saved.')
