"""Independent SSE reconstruction from previous topology audit, then final report."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
audit = pd.read_csv(HERE.parents[2]/'docs/results/task1_time80_20/neighbor_failure_analysis/structure_errors_by_regime.csv')
rows = []
for (panel, regime), group in audit.groupby(['panel', 'regime'], sort=False):
    n = int(group.n.sum())
    for method in ['LightGBM', 'Transformer', 'isolated_both']:
        sse = np.zeros(2)
        for row in group.itertuples():
            source = 'lgb' if method == 'LightGBM' or (method == 'isolated_both' and row.structure == 'isolated') else 'tf'
            sse += np.array([getattr(row, source+'_speed_sse'), getattr(row, source+'_flow_sse')])
        rmse = np.sqrt(sse/n)
        score = float(np.maximum(0, 1-rmse/np.array([25., 600.])) @ np.array([.54, .46]))
        rows.append(dict(panel=panel, regime=regime, method=method, n=n,
            rmse_speed_kmh=rmse[0], rmse_flow_per_lane=rmse[1], S_state=score))
expected = pd.DataFrame(rows)
if '--preview' in sys.argv:
    print(expected.groupby('method')[['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']].mean().to_string())
    raise SystemExit(0)

assert json.loads((OUT/'COMPLETE.json').read_text())['state'] == 'complete'
actual = pd.read_csv(OUT/'regime_metrics.csv')
joined = actual.merge(expected, on=['panel', 'regime', 'method'], suffixes=('', '_independent'), validate='one_to_one')
assert len(joined) == 90
max_error = 0.
for field in ['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']:
    difference = np.abs(joined[field]-joined[field+'_independent'])
    max_error = max(max_error, float(difference.max()))
    assert (difference < 1e-10).all(), field
joined.to_csv(OUT/'independent_audit_reconstruction.csv', index=False)
check_path = OUT/'checks.json'
check = json.loads(check_path.read_text())
check['independent_previous_structure_sse_reconstruction_max_difference'] = max_error
check_path.write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')

summary = pd.read_csv(OUT/'method_summary.csv').set_index('method')
comparison = pd.read_csv(OUT/'comparison.csv').set_index('panel')
routes = pd.read_csv(OUT/'routing_counts.csv')
ci = pd.read_csv(OUT/'paired_calendar_bootstrap.csv')
total, fallback = int(routes.n.sum()), int(routes.fallback_n.sum())
compact = pd.DataFrame(index=comparison.index)
for field, label in [('rmse_speed_kmh', 'speed_RMSE'), ('rmse_flow_per_lane', 'flow_per_lane_RMSE'), ('S_state', 'S_state')]:
    for method in ['LightGBM', 'Transformer', 'isolated_both']:
        compact[label+'__'+method] = comparison[field+'__'+method]
compact['S_state_delta_vs_Transformer'] = comparison['S_state__hybrid_minus_tf']
compact.to_csv(OUT/'ten_corridor_comparison.csv')
improved = int((compact.S_state_delta_vs_Transformer > 1e-12).sum())
speed_improved = int((comparison['rmse_speed_kmh__hybrid_minus_tf'] < -1e-12).sum())
flow_improved = int((comparison['rmse_flow_per_lane__hybrid_minus_tf'] < -1e-12).sum())
text = f'''固定“默认Transformer，仅图内孤立回退LightGBM”的最终留出评估

日期：2031-01-05—2031-02-28，共55天，十条走廊，总计{total:,}个模板目标。
模型与规则冻结：默认Transformer；节点在原模型发布节点构成的对称一/二跳图中无非自身邻居时，速度和流量都使用LightGBM。不再判断单侧时间观测缺失，也没有比较其他回退条件或选择最优走廊规则。
未训练、未重新运行模型推理，直接组合已保存的两套最终留出预测。模型文件SHA256与内部验证分析时一致。两套基线原有误差重算通过，全部550个日期缓存的目标键与预测对齐，没有重复键，预测有限非负。

回退{fallback:,}个目标，占{fallback/total:.4%}。图孤立是固定路网结构属性，不表示某个时刻邻居观测暂时全缺。

统计口径：S_state先按每条走廊R1/R2/R3等权平均，再将十条走廊等权平均；下面的RMSE使用同样的宏平均。速度单位km/h；流量单位veh/h/lane，不是全车道总流量。
speed_only或flow_only仅用于分离两个通道对固定规则的贡献，另一通道始终使用Transformer；主方案为isolated_both。

总表：
'''
text += summary.to_string()
text += '\n\n十条走廊分别比较（每走廊R1/R2/R3宏平均）：\n'+compact.to_string()
text += f'\n\n相比纯Transformer：综合得分提升{improved}条；速度误差下降{speed_improved}条；每车道流量误差下降{flow_improved}条。\n'
text += '\n回退数量：\n'+routes.to_string(index=False)
text += '\n\n整体配对日期重采样区间：\n'+ci[ci.panel == 'equal_corridor_mean'].to_string(index=False)
text += '''

解释与限制：
1. 这是按内部验证确定的固定规则在既有最终留出上的一次复核。先前已反复查看过1—2月并进行缺失与图结构诊断，因此该留出集不能称为完全未触碰测试集。
2. 日期重采样为十条走廊同步抽取55个日期、1000次，seed=2026，描述固定模型在这段日期内的波动，不包含重训和跨月份的不确定性，也没有对多走廊比较作多重校正。
3. 不按本次留出结果逐走廊改规则；某些走廊可能退化，完整保留在表中。
4. 这是本地Task 1分数，不是准确率、不是四任务总分，也未评估官方Task 3物理一致性。没有向Kaggle提交。
5. 各走廊holdout_routed_predictions.parquet仅覆盖本地55天留出目标。speed_kmh、flow_vph保留原提交单位，另附flow_per_lane和use_lightgbm以供分析，不是完整Kaggle提交文件。

主要文件：ten_corridor_comparison.csv十条完整三方案对照；method_summary.csv总体宏平均；regime_metrics.csv各R1/R2/R3；corridor_metrics.csv各走廊长表；routing_counts.csv回退覆盖；paired_calendar_bootstrap.csv配对日期重采样；checks.json校验与来源SHA256；independent_audit_reconstruction.csv独立使用此前结构分组SSE重构的90组指标。
'''
(OUT/'结果解读.txt').write_text(text, encoding='utf-8')
print('CHECK max independent error:', max_error)
print('TOTAL', total, 'FALLBACK', fallback, 'FRACTION', fallback/total)
print('IMPROVED state/speed/flow', improved, speed_improved, flow_improved)
print(summary.to_string())
print(compact.to_string())
print(ci[ci.panel == 'equal_corridor_mean'].to_string(index=False))
print('UTF-8 report saved.')
