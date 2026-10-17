"""Write a Chinese report from completed fixed-rule postprocessing."""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parent/'results'
assert json.loads((root/'status.json').read_text())['state'] == 'complete'
summary = pd.read_csv(root/'method_summary.csv').set_index('method')
comparison = pd.read_csv(root/'comparison.csv')
regime = pd.read_csv(root/'regime_metrics.csv')
channel = regime.groupby(['panel', 'method'])[['rmse_speed_kmh', 'rmse_flow_per_lane']].mean().unstack('method')
channel.columns = ['__'.join(c) for c in channel.columns]
channel.to_csv(root/'corridor_channel_comparison.csv')
counts = pd.read_csv(root/'routing_counts.csv')
counts = counts[counts.rule == 'union'].groupby('channel')[['n', 'fallback_n', 'endpoint_only_n', 'isolated_only_n', 'overlap_n']].sum()
counts['fallback_fraction'] = counts.fallback_n/counts.n
counts.to_csv(root/'total_routing_counts.csv')
ci = pd.read_csv(root/'paired_calendar_bootstrap.csv')
groups = pd.read_csv(root/'pooled_group_metrics.csv')
text = '''内部验证集固定回退规则实验

范围：十条走廊，2030-12-14 至 2031-01-04，共22天、2,477,738个模板目标。
复用已冻结的内部验证预测；没有训练、没有重新推理、没有阈值搜索。

主规则：默认Transformer。对速度、每车道流量分别判断，缺少任一侧同日可见时间观测，或原模型图内没有非自身邻居，就改用LightGBM该通道预测。
两侧都缺也回退；此处端点可见性沿用已冻结特征的同日搜索范围，不额外跨日查找。
图内孤立按原模型发布节点的一跳边并集对称化判断；孤立节点的两跳图仍没有非自身邻居。不是邻居当前时刻全缺，也没有使用候选修复图。

总表：S_state为先按每走廊R1/R2/R3等权、再按十条走廊等权平均。
下表RMSE同样为30个走廊×regime的RMSE宏平均，不是把所有目标混合计算一次RMSE。
速度单位km/h，流量单位veh/h/lane；不是总道路流量。
speed_only/flow_only指仅该通道执行回退，另一通道始终使用Transformer。

'''
text += summary.to_string()+'\n\n逐走廊S_state（内部验证，与最终留出分数不同）：\n'
text += comparison[['panel', 'LightGBM', 'Transformer', 'union_both', 'union_delta_vs_Transformer']].to_string(index=False)
text += '\n\n回退数量（两个通道独立判断，此数据上恰好相同）：\n'+counts.to_string()
text += '\n\n配对日期重采样（1000次；十条走廊同步抽日期）：\n'+ci[ci.method.isin(['union_both', 'union_speed_only', 'union_flow_only'])].to_string(index=False)
text += '\n\n分组误差（这里是按目标数汇总SSE后计算的pooled RMSE，与上表宏平均口径不同）：\n'+groups.to_string(index=False)
text += '''

解读：
1. 主规则两个通道均改善，十条中六条提升；此前最终留出下降的四条在本次内部验证上均比纯Transformer提升。
2. 仅图内孤立回退得分0.9359983650，略高于主规则0.9359805778，差0.0000177873；这是固定条件消融，不是新训练。
3. 图仍连通、仅缺少时间端点的24,654个位置，Transformer的pooled速度/流量RMSE均比LightGBM低。将它们也改用LightGBM会抵消部分收益。
4. 两个条件同时满足的7,720个位置，回退效果尤其明显，但应结合样本数和分组异质性看待。
5. 回退规则只读取输入可用性和拓扑，不使用真值选模型。真值只在固定路由后计分。

限制：本内部验证集已用于原模型检查点选择，以上属于开发期比较。日期重采样区间只描述这些冻结模型在这22天内的波动，不覆盖重新训练、其他月份或其他路网的不确定性。没有运行最终留出或官方提交评测，也不是准确率或四任务总分。

文件：method_summary.csv总表；comparison.csv逐走廊得分；corridor_channel_comparison.csv逐走廊速度/流量；regime_metrics.csv逐R1/R2/R3；pooled_group_metrics.csv条件分组；routing_counts.csv逐走廊路由计数；checks.json检查与输入SHA256。各走廊routed_predictions.parquet保存主规则结果，流量为每车道值，不能直接作为Kaggle提交。
'''
(root/'结果解读.txt').write_text(text, encoding='utf-8')
print(counts.to_string())
print('UTF-8 report saved in results directory.')
