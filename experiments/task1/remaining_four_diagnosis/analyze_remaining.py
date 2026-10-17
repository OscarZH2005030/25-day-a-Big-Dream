"""Input-defined, per-channel diagnosis of a frozen isolated-node fallback.

Existing predictions only. No model imports, fitting, inference, or route changes.
Signed group contributions exactly sum to the corridor's macro score deficit.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
TF = Path(__file__).resolve().parents[1]/'full_dataset_training_v2/runs/all_corridors_time80_20'
LGB = HERE.parent/'lightgbm_time80_20/runs/all_corridors_time80_20'
INTERNAL = HERE.parent/'internal_gap_switch/results'
PANELS = ['D12_I405_S', 'D12_I5_S', 'D7_I210_W', 'D7_I405_S']
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
CHANNELS = ['speed', 'flow_per_lane']
SCALE = [25., 600.]
WEIGHT = [.54, .46]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def describe_day(folder, names, adjacency):
    keys = pd.read_parquet(folder/'keys.parquet')
    ii, jj = np.load(folder/'ii.npy'), np.load(folder/'jj.npy')
    raw = np.load(folder/'features.npy', mmap_mode='r')[jj, ii]
    seen = np.load(folder/'visible.npy', mmap_mode='r').astype(bool)
    assert not seen[jj, ii].any()
    ticks = np.arange(288)[:, None]
    fields = []
    def feature(name):
        return raw[:, names.index(name)].astype('float64')
    relative = feature('relative_speed_initial')
    congestion = np.select([~np.isfinite(relative), relative < .5, relative < .8],
        ['unknown', 'initial_relative_speed_below0.5', 'initial_relative_speed_0.5to0.8'], default='initial_relative_speed_ge0.8')
    for ci, prefix in enumerate(['speed_kmh', 'flow_vph']):
        prev, nxt = feature(prefix+'_prev_gap_min'), feature(prefix+'_next_gap_min')
        bounded = np.isfinite(prev) & np.isfinite(nxt)
        span = np.where(bounded, prev+nxt, np.nan)
        gap = np.select([~np.isfinite(prev) & ~np.isfinite(nxt), ~bounded, span <= 20, span <= 60],
            ['both_endpoints_missing', 'one_endpoint_missing', 'bounded_span_le20', 'bounded_span20to60'], default='bounded_span_over60')
        # Count the same original 1/2-hop graph's non-self visible neighbors.
        counts = seen[:, :, ci].astype('int16') @ adjacency.T.astype('int16')
        now = counts[jj, ii] > 0
        nearby = np.lib.stride_tricks.sliding_window_view(np.pad(counts, ((6, 6), (0, 0))), 13, axis=0).max(axis=-1)
        near = nearby[jj, ii] > 0
        one = feature(prefix+'_up_count')+feature(prefix+'_down_count') > 0
        spatial = np.select([one, now, near], ['onehop_visible_now', 'only_twohop_visible_now', 'only_neighbor_within30min'], default='no_neighbor_evidence_within30min')
        seen_channel = seen[:, :, ci]
        left = np.maximum.accumulate(np.where(seen_channel, ticks, -1), axis=0)+1
        right = np.minimum.accumulate(np.where(seen_channel, ticks, 288)[::-1], axis=0)[::-1]-1
        missing_run = (right[jj, ii]-left[jj, ii]+1)*5
        run_group = np.select([missing_run == 1440, ~bounded, missing_run <= 15, missing_run <= 60],
            ['whole_day', 'day_edge_truncated', 'missing_run5to15', 'missing_run20to60'], default='missing_run_over60')
        physical_scale = feature('free_speed_kmh') if ci == 0 else feature('capacity_vph')/feature('lanes')
        endpoint_change = np.abs(feature(prefix+'_next')-feature(prefix+'_prev'))/np.maximum(physical_scale, 1e-6)
        change = np.select([~np.isfinite(endpoint_change), endpoint_change <= .1, endpoint_change <= .2],
            ['endpoint_change_unknown', 'endpoint_change_le0.1', 'endpoint_change0.1to0.2'], default='endpoint_change_over0.2')
        fields.append(pd.DataFrame(dict(gap=gap, missing_run=run_group, spatial=spatial,
            initial_speed=congestion, endpoint_change=change, endpoint_span_min=span,
            prev_gap_min=prev, next_gap_min=nxt)))
    return keys, ii, jj, fields


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise SystemExit('Preserve completed diagnosis.')
    plan = json.loads((TF/'plan.json').read_text())
    save_json('plan.json', dict(panels=PANELS, phases=['holdout', 'internal'], training=False,
        inference=False, routing_changes=False, route='Transformer unless original graph isolated then LightGBM',
        primary_gap='two finite visible endpoint gaps sum >60 min is long; <=20 and (20,60] descriptive subgroups',
        missing_run='separate descriptive consecutive missing-slot duration; day-edge truncated',
        spatial='same original symmetrized 1/2-hop graph, excluding self, same time and +/-30min same day',
        label_independent_groups=True, hints='initial_relative_speed uses interpolation/history, not true congestion',
        score_attribution='positive means LightGBM S_state minus hybrid; exact signed marginal RMSE decomposition, R1/R2/R3 macro',
        caveat='holdout previously inspected; descriptive diagnosis, not causal proof or route tuning', script_sha256=sha(Path(__file__))))
    group_tables, summaries, contributor_tables, source_audits = [], [], [], []
    totals = {}
    for phase in ['holdout', 'internal']:
        dates = plan['holdout_dates' if phase == 'holdout' else 'internal_validation_dates']
        for panel in PANELS:
            meta = json.loads((TF/panel/'cache/ready.json').read_text())
            with np.load(TF/panel/'history.npz', allow_pickle=False) as history:
                links = history['links'].tolist()
            topo = pd.read_csv(Path(plan['release_root'])/'corridors'/panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
            adj = np.zeros((len(links), len(links)), dtype=bool)
            for i, link in enumerate(links):
                others = str(topo.loc[link, 'incoming_link_ids']).split(';')+str(topo.loc[link, 'outgoing_link_ids']).split(';')
                adj[i] = [other in others for other in links]
            adj |= adj.T.copy()
            np.fill_diagonal(adj, False)
            isolated_nodes = ~adj.any(axis=1)
            adjacency = adj | ((adj.astype('int16') @ adj.astype('int16')) > 0)
            np.fill_diagonal(adjacency, False)
            assert np.array_equal(~adjacency.any(axis=1), isolated_nodes)
            if phase == 'holdout':
                tfpath, lgpath = TF/panel/'holdout_predictions.parquet', LGB/panel/'holdout_predictions.parquet'
                p, l = pd.read_parquet(tfpath), pd.read_parquet(lgpath)
                assert p[KEYS].astype(str).equals(l[KEYS].astype(str))
                fd = pd.read_csv(Path(plan['release_root'])/'corridors'/panel/'network/fd_parameters.csv').set_index('link_id')
                lanes = p.link_id.map(fd.lanes).to_numpy()
                tf = np.column_stack([p.speed_kmh, p.flow_vph/lanes])
                lg = np.column_stack([l.speed_kmh, l.flow_vph/lanes])
                sources = [tfpath, lgpath]
            else:
                ppath = INTERNAL/panel/'internal_predictions.parquet'
                p = pd.read_parquet(ppath)
                tf = np.column_stack([p[c+'_tf'] for c in CHANNELS])
                lg = np.column_stack([p[c+'_lgb'] for c in CHANNELS])
                sources = [ppath]
            assert not p.duplicated(KEYS).any()
            cursor, chunks = 0, [[], []]
            for di, date in enumerate(dates):
                folder = TF/panel/'cache'/date
                keys, ii, jj, fields = describe_day(folder, meta['names'], adjacency)
                end = cursor+len(keys)
                assert p[KEYS].iloc[cursor:end].reset_index(drop=True).astype(str).equals(keys[KEYS].astype(str))
                truth = np.load(folder/'label.npy', mmap_mode='r')[jj, ii].astype('float64')*np.array(SCALE)
                base = np.load(folder/'base.npy', mmap_mode='r')[jj, ii].astype('float64')*np.array(SCALE)
                iso = isolated_nodes[ii]
                for ci, channel in enumerate(CHANNELS):
                    f = fields[ci]
                    f['panel'], f['phase'], f['channel'] = panel, phase, channel
                    f['date'], f['regime'] = date, keys.mask_regime.to_numpy()
                    f['station_id'], f['link_id'] = keys.station_id.to_numpy(), keys.link_id.to_numpy()
                    f['isolated'] = iso
                    f['n'] = 1
                    pred_tf, pred_lg, y = tf[cursor:end, ci], lg[cursor:end, ci], truth[:, ci]
                    hybrid = np.where(iso, pred_lg, pred_tf)
                    f['hybrid_sse'], f['lgb_sse'] = (hybrid-y)**2, (pred_lg-y)**2
                    f['hybrid_error'], f['lgb_error'] = hybrid-y, pred_lg-y
                    f['excess_sse'] = f.hybrid_sse-f.lgb_sse
                    f['positive_excess_sse'] = np.maximum(f.excess_sse, 0)
                    f['negative_excess_sse'] = np.minimum(f.excess_sse, 0)
                    f['hybrid_abs_correction_from_interpolation'] = np.abs(hybrid-base[:, ci])
                    f['lgb_abs_correction_from_interpolation'] = np.abs(pred_lg-base[:, ci])
                    f['hybrid_lgb_abs_disagreement'] = np.abs(hybrid-pred_lg)
                    chunks[ci].append(f)
                cursor = end
                if (di+1) % 20 == 0:
                    print(phase, panel, f'{di+1}/{len(dates)} days', flush=True)
            assert cursor == len(p)
            totals[phase] = totals.get(phase, 0)+len(p)
            for ci, channel in enumerate(CHANNELS):
                f = pd.concat(chunks[ci], ignore_index=True)
                f['score_deficit'] = 0.
                for regime, ids in f.groupby('regime').groups.items():
                    nh = len(ids)
                    h = float(np.sqrt(f.loc[ids, 'hybrid_sse'].sum()/nh))
                    b = float(np.sqrt(f.loc[ids, 'lgb_sse'].sum()/nh))
                    assert h < SCALE[ci] and b < SCALE[ci], 'Clipped score needs a different attribution'
                    factor = WEIGHT[ci]/(3*SCALE[ci]*nh*(h+b))
                    f.loc[ids, 'score_deficit'] = f.loc[ids, 'excess_sse']*factor
                    summaries.append(dict(phase=phase, panel=panel, channel=channel, regime=regime,
                        n=nh, hybrid_rmse=h, lgb_rmse=b, hybrid_minus_lgb_rmse=h-b,
                        score_deficit=float(f.loc[ids, 'score_deficit'].sum())))
                    np.testing.assert_allclose(f.loc[ids, 'score_deficit'].sum(), WEIGHT[ci]*(h-b)/(3*SCALE[ci]), rtol=0, atol=1e-12)
                assert (f.loc[f.isolated, 'excess_sse'] == 0).all()
                assert (f.loc[f.isolated, 'score_deficit'] == 0).all()
                f.to_parquet(OUT/(phase+'_'+panel+'_'+channel+'_diagnostics.parquet'), index=False)
                sums = ['n', 'hybrid_sse', 'lgb_sse', 'excess_sse', 'positive_excess_sse', 'negative_excess_sse',
                    'hybrid_error', 'lgb_error', 'score_deficit', 'hybrid_abs_correction_from_interpolation',
                    'lgb_abs_correction_from_interpolation', 'hybrid_lgb_abs_disagreement']
                for scope, selected in [('all', f), ('connected', f[~f.isolated]), ('isolated', f[f.isolated])]:
                    for dim in ['gap', 'missing_run', 'spatial', 'initial_speed', 'endpoint_change', 'regime']:
                        g = selected.groupby(dim, as_index=False, dropna=False)[sums].sum().rename(columns={dim:'group'})
                        for field, value in [('scope', scope), ('dimension', dim), ('phase', phase), ('panel', panel), ('channel', channel)]:
                            g[field] = value
                        g['hybrid_rmse'], g['lgb_rmse'] = np.sqrt(g.hybrid_sse/g.n), np.sqrt(g.lgb_sse/g.n)
                        g['n_share_of_corridor'] = g.n/len(f)
                        g['hybrid_sse_share_of_corridor'] = g.hybrid_sse/f.hybrid_sse.sum()
                        g['hybrid_mean_error'], g['lgb_mean_error'] = g.hybrid_error/g.n, g.lgb_error/g.n
                        g['n_share_of_scope'] = g.n/max(1, len(selected))
                        assert abs(g.score_deficit.sum()-selected.score_deficit.sum()) < 1e-12
                        group_tables.append(g)
                selected = f[~f.isolated]
                joint = selected.groupby(['gap', 'spatial', 'initial_speed'], as_index=False, dropna=False)[sums].sum()
                joint['phase'], joint['panel'], joint['channel'], joint['dimension'] = phase, panel, channel, 'joint'
                joint['hybrid_rmse'], joint['lgb_rmse'] = np.sqrt(joint.hybrid_sse/joint.n), np.sqrt(joint.lgb_sse/joint.n)
                joint['n_share_of_corridor'] = joint.n/len(f)
                contributor_tables.append(joint)
                station = selected.groupby(['station_id', 'link_id'], as_index=False)[sums].sum()
                station['phase'], station['panel'], station['channel'], station['dimension'] = phase, panel, channel, 'station'
                contributor_tables.append(station)
            source_audits.append(dict(phase=phase, panel=panel, n=len(p), sources=[dict(path=str(s), sha256=sha(s)) for s in sources]))
            print(phase, panel, 'COMPLETE', len(p), flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'channel_by_regime.csv', index=False)
    macro = summary.groupby(['phase', 'panel', 'channel'], as_index=False).agg(
        n=('n', 'sum'), hybrid_rmse=('hybrid_rmse', 'mean'), lgb_rmse=('lgb_rmse', 'mean'),
        hybrid_minus_lgb_rmse=('hybrid_minus_lgb_rmse', 'mean'), score_deficit=('score_deficit', 'sum'))
    macro.to_csv(OUT/'channel_summary.csv', index=False)
    reference = pd.read_csv(HERE.parent/'holdout_isolated_fallback/results/comparison.csv').set_index('panel')
    for row in macro[macro.phase == 'holdout'].itertuples():
        name = 'rmse_speed_kmh' if row.channel == 'speed' else 'rmse_flow_per_lane'
        assert abs(row.hybrid_rmse-reference.loc[row.panel, name+'__isolated_both']) < 1e-10
        assert abs(row.lgb_rmse-reference.loc[row.panel, name+'__LightGBM']) < 1e-10
    groups = pd.concat(group_tables, ignore_index=True)
    groups.to_csv(OUT/'input_group_attribution.csv', index=False)
    contributors = pd.concat(contributor_tables, ignore_index=True)
    contributors.to_csv(OUT/'joint_and_station_attribution.csv', index=False)
    for phase in ['holdout', 'internal']:
        top = contributors[(contributors.phase == phase)&(contributors.dimension == 'joint')]
        top = top.sort_values('score_deficit', ascending=False).groupby(['panel', 'channel'], sort=False).head(8)
        top.to_csv(OUT/(phase+'_top_joint_cases.csv'), index=False)
    save_json('checks.json', dict(training_started=False, inference_started=False, rule_changed=False,
        targets=totals, target_keys='all cached date keys match frozen predictions',
        isolated_deficit='exactly zero in both channels',
        group_attribution='every disjoint dimension sums to exact corridor channel macro score deficit, tolerance1e-12',
        heldout_channel_metrics_match_previous_tolerance=1e-10, inputs=source_audits))
    save_json('COMPLETE.json', dict(state='complete', panels=PANELS, targets=totals))
    print('\nCHANNEL SUMMARY\n'+macro.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
