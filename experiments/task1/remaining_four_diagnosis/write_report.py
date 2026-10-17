"""Write evidence and a prospective residual design; no model fitting."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE/'results'
assert json.loads((ROOT/'COMPLETE.json').read_text())['state'] == 'complete'
s = pd.read_csv(ROOT/'channel_summary.csv')
g = pd.read_csv(ROOT/'input_group_attribution.csv')
g = g[g.scope == 'connected']
selections = [
    ('D12_I405_S', 'speed', 'bounded_span_le20'),
    ('D12_I5_S', 'speed', 'bounded_span_le20'),
    ('D7_I405_S', 'speed', 'bounded_span_le20'),
    ('D7_I210_W', 'flow_per_lane', 'bounded_span_le20'),
    ('D7_I210_W', 'flow_per_lane', 'one_endpoint_missing'),
    ('D7_I405_S', 'flow_per_lane', 'one_endpoint_missing'),
]
cases = []
for phase in ['holdout', 'internal']:
    for panel, channel, group in selections:
        row = g[(g.phase == phase)&(g.panel == panel)&(g.channel == channel)&(g.dimension == 'gap')&(g.group == group)].iloc[0]
        total = s[(s.phase == phase)&(s.panel == panel)&(s.channel == channel)].iloc[0]
        cases.append(dict(phase=phase, panel=panel, channel=channel, gap=group, n=int(row.n),
            group_pooled_hybrid_rmse=row.hybrid_rmse, group_pooled_lgb_rmse=row.lgb_rmse,
            n_share_corridor=row.n_share_of_corridor, score_deficit_contribution=row.score_deficit,
            whole_channel_score_deficit=total.score_deficit,
            contribution_over_net_channel_deficit=row.score_deficit/total.score_deficit))
cases = pd.DataFrame(cases)
cases.to_csv(ROOT/'key_missing_cases.csv', index=False)

# Independently compare the per-channel and net deficit with earlier scored runs.
max_difference = 0.
for phase, reference_folder in [('holdout', HERE.parent/'holdout_isolated_fallback/results'),
                                 ('internal', HERE.parent/'internal_fallback_switch/results')]:
    reference = pd.read_csv(reference_folder/'regime_metrics.csv')
    reference = reference.groupby(['panel', 'method'])[['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']].mean()
    for (panel, channel), a in s[s.phase == phase].groupby(['panel', 'channel']):
        field = 'rmse_speed_kmh' if channel == 'speed' else 'rmse_flow_per_lane'
        for method, actual in [('isolated_both', a.hybrid_rmse.iloc[0]), ('LightGBM', a.lgb_rmse.iloc[0])]:
            difference = abs(reference.loc[(panel, method), field]-actual)
            assert difference < 1e-10
            max_difference = max(max_difference, difference)
    for panel, a in s[s.phase == phase].groupby('panel'):
        expected = reference.loc[(panel, 'LightGBM'), 'S_state']-reference.loc[(panel, 'isolated_both'), 'S_state']
        difference = abs(a.score_deficit.sum()-expected)
        assert difference < 1e-12
        max_difference = max(max_difference, difference)
checks = json.loads((ROOT/'checks.json').read_text())
checks['both_phase_metrics_and_score_deficit_independently_match_max_error'] = max_difference
(ROOT/'checks.json').write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding='utf-8')

report = '''四条走廊：孤立点回退后的剩余误差诊断

范围与方法
本次复用四条走廊已有预测，覆盖最终留出55天的1,391,517个目标，并在内部验证22天的541,892个目标上复核分组方向。未训练、未重新推理、未修改固定回退规则。
孤立节点的组合预测等于LightGBM，其超额误差与分差贡献严格为零。剩余净分差来自534,609个连通节点。

整体通道误差采用每走廊R1/R2/R3的RMSE宏平均；局部缺失组使用组内全部目标的pooled RMSE。二者统计口径不同，不能用局部RMSE差直接推断官方宏分贡献。
分差贡献则精确分解原宏分：对每个走廊、R组、通道，组内差平方误差和乘以
weight / [3 * scale * N_R * (RMSE_hybrid_R + RMSE_lgb_R)]。
正值表示该组使LightGBM领先，负值表示该组使组合领先。每个互斥维度的各组贡献可精确加总成通道分差；不同维度的贡献不能再相加。
净分差是有正有负的和，因此某坏组贡献超过净分差100%意味着其他组抵消了一部分，并不表示统计错误。所有计分通道均未触发零分截断，公式适用。

一、四条走廊的通道差距
速度单位km/h；流量单位veh/h/lane。hybrid_minus_lgb_rmse为正才表示组合落后。
'''
report += s[s.phase == 'holdout'].to_string(index=False)
report += '''

D12_I405_S和D12_I5_S：流量已经更好，净落后主要来自速度。
D7_I210_W：速度已经略好，净落后来自流量。
D7_I405_S：速度和流量均仍有差距。

二、剩余差距最集中的缺失情况
这里“短缺口”细分为前后两条可见观测时间间隔<=20分钟，对规则5分钟网格通常对应连续缺失5—15分钟。它不是预测未来20分钟。长缺口仍统一为双侧可见端点时间间隔>60分钟。
单侧端点缺失由原特征的同日搜索确定，可能位于日界；不等价于真实时间线上永远没有另一侧观测。此数据这四条的目标中没有整日双侧端点都缺的组。
'''
report += cases[cases.phase == 'holdout'].to_string(index=False)
report += '''

主要读数：
1. D12_I405_S的连通短缺口速度：54,577个，组合/LGB RMSE 1.864/1.778；贡献约102%的速度净落后。
2. D12_I5_S：164,910个，2.066/1.977；贡献约96%的速度净落后。
3. D7_I405_S：88,175个，2.796/2.548；贡献约127%的速度净落后，较长缺口抵消一部分。
4. D7_I210_W流量：短缺口87,361个，33.821/32.668，贡献约81%的流量净落后；单侧缺失1,551个，46.867/31.715，另贡献约25%，其他组有抵消。
5. D7_I405_S流量：单侧缺失1,507个，76.501/48.251，贡献约104%的流量净落后；双侧20—60分钟组反而帮助组合追回分数。

三、速度大误差与空间证据
四条连通目标中，初始相对速度initial/free_speed在[0.5,0.8)的只有3,316个，占0.62%，却贡献49.3%的连通速度平方误差。
该数值来自插值/历史初值，是输入提示，不是真实拥堵标签，不能据此认定全是拥堵波。
例如D7_I405_S中，短缺口、当前邻居无观测但前后30分钟有、初始相对速度[0.5,0.8)这一联合组只有146个目标，速度RMSE达到43.875/34.622。
这说明少量高误差状态对结果影响很大，不能只看整体缺失率。

完全没有前后30分钟邻居证据的连通目标只有1,199个，占连通目标0.22%，并非主要损失来源。
相反，“当前邻居无观测、但前后30分钟有”的210,152个目标，速度RMSE 3.135/2.926，存在可利用但未必正确对齐的信息。
在D12_I405_S，一跳当前可见组仍贡献多数速度净分差；在D7_I210_W，一跳当前可见组仍有流量损失。因此“邻居可见”也不能等同于“图分支可靠”。
这些是关联证据；尚不能从现有结果证明具体是传播时延、注意力权重、容量瓶颈或训练覆盖不足所致。

四、内部验证复核与稳定性
'''
report += s[s.phase == 'internal'].to_string(index=False)
report += '''

D7_I210_W的短缺口/单侧流量问题、D7_I405_S的单侧流量问题在内部验证也出现。
速度则明显更不稳定：D12_I5_S和D7_I405_S的组合速度在内部验证总体优于LightGBM，最终留出反而略差；[0.5,0.8)初值组也有模型优劣反转。
不能根据这次最终留出，追加“短缺口一律回退”或“中间初速一律回退”的硬规则，更不能把留出期最差的站点ID设为回退名单。

五、源码支持的瓶颈
当前Transformer已经是“时间插值 + 神经网络残差”，输出层已有零初始化；仅再加一个残差头或重复零初始化不是新的改进。
其图注意力neighbor_valid只处理图邻域/padding可用性，没有显式按各邻居观测年龄和通道可靠性给消息收缩权重；模型可通过输入特征隐式学习，但当前没有专门的可靠性门控。
时间窗口24步；prev/next、offset和插值均按日构建。单侧端点缺失时初值回退历史，日界可能损失离线任务本来允许使用的相邻日期发布观测。
这些实现事实与上面的短缺口、单侧流量和异步邻居现象相容，但改哪一处有效仍需独立消融。

六、建议的下一版残差方案（设计，未实现训练）
以LightGBM的每通道预测b_c为基底，而非插值：
  yhat_c = nonnegative(b_c + g_c(z) * delta_c(context))
  c = speed 或 flow_per_lane；两个通道独立门控、独立残差头、独立修正正则。
第一轮仍保留原图；孤立节点按已冻结规则直接使用LightGBM。连通节点允许学习小修正。
51基础特征和51掩码继续使用，加入b_c及其相对插值/历史的偏差，以及明确的观测年龄/可用性摘要。这里是新方案的输入扩展，不能再称作原102维完全不变的对照。

速度头：使用小型双向时序残差模块学习基线在局部变化中的偏差。门控读取双侧距离、端点变化、初始相对速度、邻居可见性和时间偏移，不把“短缺口”直接当成必回退条件。
流量头：优先学习短缺口的细小偏差与单侧缺失时的基线修正，低证据时允许修正收缩到零；独立于速度决策，避免破坏D12两条已经改善的流量。
空间消息：对每个通道使用当前/一二跳/窗口内可见比例及最近观测年龄作软可靠性权重。邻居当前缺值时仍可携带时间线索，不因同刻缺失就一律硬屏蔽。对有证据但不同步的邻居学习时间偏移，不能把路径跳数伪称为传播分钟。
跨日上下文：作为单独消融，用真实存在且已发布的前后日期观测更新prev/next、offset和窗口halo；禁止将未遮蔽真值、监督标签或本次误差分组结果放入输入。
头部保留零初始化，使新网络从LightGBM开始；增加适度修正收缩正则，幅度与强度只在内部验证决定，不能据本次留出手设。不会预期一个小残差必定解决所有高误差尾部。

七、防泄漏训练安排（尚未执行）
使用168个监督训练日内的时间块交叉拟合LightGBM，例如4个连续42天折：每折b_c由没有使用该折标签的树产生，再以y-b_c训练残差网络。不能使用LightGBM对自己训练标签的原位拟合残差。
折内早停/选择只使用其余监督日期中的内部分割，不能使用被预测折或最终55天留出标签。历史统计保持规定28天；需要拟合的归一化只用相应训练折/监督训练日期。
本方案不使用原Transformer训练集拟合预测作为元特征，因此不需要为此额外制造有泄漏风险的Transformer原位预测。
最终LightGBM基底按既定168天重新拟合；22天内部验证统一选残差幅度和早停，检查OOF基底与最终基底的分布差异。
稀有状态可用于分层采样提高覆盖，但对采样权重作校正并单独报告R1/R2/R3，不能偷偷改变评分目标。保持原监督权重和零物理系数作为首轮对照。

最低成本验证顺序：
A. 当前已冻结孤立回退组合，作为参照。
B. LightGBM基底 + 小型双通道门控残差，保留原图和当日上下文；先验证“换基底/保留基线”的增益。
C. 在B上仅增加跨日已发布上下文，检查单侧流量组。
D. 在B上仅增加邻居年龄/时间对齐的软消息权重，检查当前缺失但临近有证据的速度组。
候选压缩图作为后续单独消融；当前剩余分差已位于连通节点，不能把修图当作全部解释。匝道和物理损失继续按后续独立实验安排。
在失败走廊之外保留一个原本有提升的走廊作对照，防止只为这四条适配；各项选择以内部验证为准。

验收：速度和每车道流量分别看宏RMSE及R1/R2/R3，检查短缺口、单侧缺失、初始中间速度、异步邻居、高误差站点与修正幅度分布。最终留出已经用于多次诊断，此后其上的改进属于探索性复核，不再称为未触碰测试证明。

输出：channel_summary.csv整体通道；key_missing_cases.csv主要缺失组；input_group_attribution.csv互斥分组精确归因；problem_channel_top_cases.csv联合组；top_station_contributions.csv站点集中情况；checks.json校验与来源SHA256。
'''
(ROOT/'四条走廊剩余误差与残差方案.txt').write_text(report, encoding='utf-8')
print('REPORT SAVED; max independent metric difference:', max_difference)
print(cases[cases.phase == 'holdout'].to_string(index=False))
