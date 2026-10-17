"""Summarize the fixed-rule internal-validation inference; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).parent/'results'
if json.loads((root/'status.json').read_text())['state']!='complete':
    raise SystemExit('Wait until all ten inference evaluations complete.')
c=pd.read_csv(root/'all_channel_metrics.csv')
macro=c[(c.gap_group=='all')&(c.regime.isin(['R1','R2','R3']))]
channel=macro.groupby(['channel','method'],as_index=False)[['rmse','S_channel']].mean()
channel=channel.rename(columns={'rmse':'mean_corridor_regime_RMSE','S_channel':'mean_corridor_regime_S_channel'})
channel.to_csv(root/'channel_summary.csv',index=False)
pooled=[]
for (channel_name,gap,method),a in c[c.regime=='pooled'].groupby(['channel','gap_group','method']):
    pooled.append(dict(channel=channel_name,gap_group=gap,method=method,n=int(a.n.sum()),
                       pooled_rmse=float(np.sqrt(np.sum(a.n*a.rmse**2)/a.n.sum()))))
pd.DataFrame(pooled).to_csv(root/'pooled_group_metrics.csv',index=False)
methods=pd.read_csv(root/'all_metrics.csv')
overall=methods[methods.regime=='macro_mean'].groupby('method',as_index=False).S_state.mean()
baseline=float(overall[overall.method=='LightGBM'].S_state.iloc[0])
overall['delta_vs_LightGBM']=overall.S_state-baseline
overall.to_csv(root/'method_summary.csv',index=False)
comparison=pd.read_csv(root/'comparison.csv')
np.testing.assert_allclose(comparison.hybrid_both-comparison.LightGBM,
    comparison.hybrid_speed_only+comparison.hybrid_flow_only-2*comparison.LightGBM,rtol=0,atol=1e-12)

route_rows=[]
for folder in sorted(root.iterdir()):
    if not folder.is_dir() or not (folder/'COMPLETE.json').exists():continue
    p=pd.read_parquet(folder/'internal_predictions.parquet')
    for field in ['speed','flow_per_lane']:
        span=p[field+'_endpoint_span_min'].to_numpy()
        new=np.isfinite(span)&(span>60)
        old=np.isfinite(span)&((span-5)>60)
        assert np.all(span[new&~old]==65)
        assert np.array_equal(new,p[field+'_use_tf'].to_numpy())
        assert np.isfinite(p[field+'_hybrid']).all()
        expected=np.where(new,p[field+'_tf'],p[field+'_lgb'])
        np.testing.assert_array_equal(expected,p[field+'_hybrid'].to_numpy())
        route_rows.append(dict(panel=folder.name,channel=field,n=len(p),new_long=int(new.sum()),
                               old_long=int(old.sum()),added_exact_span65=int((new&~old).sum())))
pd.DataFrame(route_rows).to_csv(root/'definition_alignment_counts.csv',index=False)
checks=[json.loads((root/panel/'COMPLETE.json').read_text()) for panel in pd.read_csv(root/'comparison.csv').panel]
audit=dict(training_started=False,threshold_minutes=60,strictly_greater=True,
           missing_endpoint_falls_back_to='LightGBM',gap_search_scope='same day, matching frozen feature cache',
           total_internal_targets=sum(r['n'] for r in checks),
           max_transformer_internal_score_replay_error=max(abs(r['transformer_score_difference']) for r in checks),
           exact_switch_rule_check='passed for both channels and all targets',
           independent_channel_score_additivity='passed within 1e-12',
           caveat='Same internal development dates used for checkpoint selection; no claim of untouched test performance.')
(root/'checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('METHODS\n'+overall.to_string(index=False))
print('CHANNELS\n'+channel.to_string(index=False))
print('COUNTS\n'+pd.DataFrame(route_rows).groupby('channel')[['n','new_long','old_long','added_exact_span65']].sum().to_string())
print('LONG GROUP\n'+pd.DataFrame(pooled).query("gap_group == 'span_over60min'").to_string(index=False))
print('CHECKS',audit)
