"""Aggregate the paired, fixed-rule diagnosis without fitting or selecting rules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent/'results'
assert json.loads((ROOT/'COMPLETE.json').read_text())['state'] == 'complete'
g = pd.read_csv(ROOT/'input_group_attribution.csv')
connected = g[g.scope == 'connected']
pooled = connected.groupby(['phase', 'channel', 'dimension', 'group'], as_index=False)[['n', 'hybrid_sse', 'lgb_sse', 'score_deficit']].sum()
pooled['hybrid_rmse'], pooled['lgb_rmse'] = np.sqrt(pooled.hybrid_sse/pooled.n), np.sqrt(pooled.lgb_sse/pooled.n)
pooled['connected_n_share'] = pooled.n/pooled.groupby(['phase', 'channel', 'dimension']).n.transform('sum')
pooled['connected_hybrid_sse_share'] = pooled.hybrid_sse/pooled.groupby(['phase', 'channel', 'dimension']).hybrid_sse.transform('sum')
pooled['four_corridor_mean_score_deficit'] = pooled.score_deficit/4
pooled.to_csv(ROOT/'four_corridor_connected_groups.csv', index=False)
joint = pd.read_csv(ROOT/'joint_and_station_attribution.csv')
top_stations = joint[joint.dimension == 'station'].sort_values('score_deficit', ascending=False).groupby(['phase', 'panel', 'channel']).head(5)
top_stations.to_csv(ROOT/'top_station_contributions.csv', index=False)
summary = pd.read_csv(ROOT/'channel_summary.csv')
hold = summary[summary.phase == 'holdout']
print('HOLDOUT CHANNELS\n'+hold.to_string(index=False))
print('INTERNAL CHANNELS\n'+summary[summary.phase == 'internal'].to_string(index=False))
cols = ['phase', 'channel', 'dimension', 'group', 'n', 'connected_n_share', 'hybrid_rmse', 'lgb_rmse', 'connected_hybrid_sse_share', 'four_corridor_mean_score_deficit']
print('POOLED CONNECTED\n'+pooled[pooled.dimension.isin(['gap', 'spatial', 'initial_speed'])][cols].to_string(index=False))
problem = hold[hold.score_deficit > 0][['panel', 'channel']]
top = joint[(joint.phase == 'holdout')&(joint.dimension == 'joint')].merge(problem, on=['panel', 'channel'])
top = top.sort_values('score_deficit', ascending=False).groupby(['panel', 'channel']).head(4)
print('TOP PROBLEM CASES\n'+top[['panel', 'channel', 'gap', 'spatial', 'initial_speed', 'n', 'hybrid_rmse', 'lgb_rmse', 'score_deficit']].to_string(index=False))
top.to_csv(ROOT/'problem_channel_top_cases.csv', index=False)
for phase in ['holdout', 'internal']:
    for panel, channel in problem.itertuples(index=False):
        selected = connected[(connected.phase == phase)&(connected.panel == panel)&(connected.channel == channel)]
        for dimension in ['gap', 'spatial', 'initial_speed']:
            block = selected[selected.dimension == dimension].sort_values('score_deficit', ascending=False)
            print('DETAIL', phase, panel, channel, dimension)
            print(block[['group', 'n', 'n_share_of_corridor', 'hybrid_rmse', 'lgb_rmse', 'hybrid_sse_share_of_corridor', 'score_deficit']].to_string(index=False))
