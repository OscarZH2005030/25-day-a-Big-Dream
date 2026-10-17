"""Frozen internal predictions: default Transformer, fixed input-only fallback.

No torch/lightgbm imports, fitting, inference, threshold search or held-out labels.
Run with the project's existing Python environment from any working directory.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
SOURCE = HERE.parent / 'internal_gap_switch' / 'results'
RELEASE = Path(__file__).resolve().parents[3]/'data/kaggle_public'
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
CHANNELS = ['speed', 'flow_per_lane']
REGIMES = ['R1', 'R2', 'R3']
SCALES = np.array([25., 600.])
RULES = ['endpoint_missing', 'isolated', 'union']
METHODS = ['LightGBM', 'Transformer'] + [r+'_'+c for r in RULES for c in ['both', 'speed_only', 'flow_only']]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def score_sse(sse, n):
    rmse = np.sqrt(sse / n[..., None, None])
    components = np.maximum(0, 1 - rmse / SCALES)
    return rmse, components @ np.array([.54, .46])


def main():
    assert json.loads((SOURCE/'status.json').read_text())['state'] == 'complete'
    plan = json.loads((SOURCE/'plan.json').read_text())
    dates, panels = plan['dates'], plan['panels']
    assert dates == list(pd.date_range('2030-12-14', '2031-01-04').strftime('%Y-%m-%d'))
    OUT.mkdir(exist_ok=True)
    write_json('plan.json', dict(dates=dates, panels=panels, source=str(SOURCE),
        default='Transformer', fallback='LightGBM',
        primary_rule='missing either temporal endpoint OR no non-self neighbor in original model graph',
        endpoint_scope='same day, matching frozen features; includes both endpoints missing',
        graph='original released-node graph, symmetrized 1/2-hop adjacency; excludes self',
        training=False, inference=False, threshold_tuning=False,
        script_sha256=sha(Path(__file__)), methods=METHODS,
        aggregation='mean of 3 regime scores per corridor, then equal mean of 10 corridors',
        caveat='Internal dates already used for checkpoint selection; development analysis.'))
    counts = np.zeros((len(dates), len(panels), 3), dtype=np.int64)
    cube = np.zeros((*counts.shape, len(METHODS), 2))
    route_rows, group_rows, audits, daily_rows = [], [], [], []
    topology_reference = pd.read_csv(HERE.parent/'neighbor_failure_analysis/topology_audit.csv').set_index('panel')
    for pi, panel in enumerate(panels):
        path = SOURCE/panel/'internal_predictions.parquet'
        frame = pd.read_parquet(path)
        assert len(frame) and not frame.duplicated(KEYS).any()
        assert set(frame.panel) == {panel}
        day = frame.timestamp.astype(str).str[:10]
        assert set(day) == set(dates) and set(frame.mask_regime) == set(REGIMES)
        topology_path = RELEASE/'corridors'/panel/'network/lwr_mainline_topology.csv'
        top = pd.read_csv(topology_path)
        links = set(top.mainline_link_id)
        assert not top.mainline_link_id.duplicated().any() and frame.link_id.isin(links).all()
        neighbors = {link: set() for link in links}
        for row in top.itertuples():
            for link in str(row.incoming_link_ids).split(';') + str(row.outgoing_link_ids).split(';'):
                if link in links and link != row.mainline_link_id:
                    neighbors[row.mainline_link_id].add(link)
                    neighbors[link].add(row.mainline_link_id)
        isolated_links = {link for link in links if not neighbors[link]}
        assert len(links) == topology_reference.loc[panel, 'nodes']
        assert len(isolated_links) == topology_reference.loc[panel, 'isolated']
        isolated = frame.link_id.isin(isolated_links).to_numpy()
        missing = np.column_stack([~np.isfinite(frame[c+'_endpoint_span_min'].to_numpy()) for c in CHANNELS])
        masks = {'endpoint_missing': missing, 'isolated': np.repeat(isolated[:, None], 2, axis=1),
                 'union': missing | isolated[:, None]}
        truth, lgb, tf = [np.column_stack([frame[c+'_'+suffix].to_numpy() for c in CHANNELS])
                          for suffix in ['truth', 'lgb', 'tf']]
        assert all(np.isfinite(x).all() for x in [truth, lgb, tf])
        predictions = {'LightGBM': lgb, 'Transformer': tf}
        for rule, mask in masks.items():
            both = np.where(mask, lgb, tf)
            predictions[rule+'_both'] = both
            predictions[rule+'_speed_only'] = np.column_stack([both[:, 0], tf[:, 1]])
            predictions[rule+'_flow_only'] = np.column_stack([tf[:, 0], both[:, 1]])
        primary = predictions['union_both']
        assert (primary >= 0).all()
        selected = frame[KEYS].copy()
        selected['graph_isolated'] = isolated
        for ci, channel in enumerate(CHANNELS):
            selected[channel+'_endpoint_missing'] = missing[:, ci]
            selected[channel+'_use_lightgbm'] = masks['union'][:, ci]
            selected[channel+'_prediction'] = primary[:, ci]
            for rule in RULES:
                mask = masks[rule][:, ci]
                np.testing.assert_array_equal(predictions[rule+'_both'][:, ci], np.where(mask, lgb[:, ci], tf[:, ci]))
                route_rows.append(dict(panel=panel, channel=channel, rule=rule, n=len(frame),
                    fallback_n=int(mask.sum()), fallback_fraction=float(mask.mean()),
                    endpoint_only_n=int((missing[:, ci] & ~isolated).sum()),
                    isolated_only_n=int((~missing[:, ci] & isolated).sum()),
                    overlap_n=int((missing[:, ci] & isolated).sum())))
            groups = {'all': np.ones(len(frame), dtype=bool),
                'endpoint_only': missing[:, ci] & ~isolated,
                'isolated_only': isolated & ~missing[:, ci],
                'both_conditions': missing[:, ci] & isolated,
                'neither': ~missing[:, ci] & ~isolated}
            assert sum(int(mask.sum()) for g, mask in groups.items() if g != 'all') == len(frame)
            for group, mask in groups.items():
                for method in ['LightGBM', 'Transformer', 'union_both']:
                    error = predictions[method][mask, ci]-truth[mask, ci]
                    group_rows.append(dict(panel=panel, channel=channel, group=group, method=method,
                        n=int(mask.sum()), sse=float(error@error)))
        selected.to_parquet(OUT/(panel+'_routed_predictions.parquet'), index=False)
        values = pd.DataFrame({'date': day, 'regime': frame.mask_regime, 'n': 1})
        for mi, method in enumerate(METHODS):
            error = predictions[method] - truth
            for ci, channel in enumerate(CHANNELS):
                values[method+'_'+channel+'_sse'] = error[:, ci]**2
        sums = values.groupby(['date', 'regime'], as_index=False).sum()
        sums['panel'] = panel
        daily_rows.append(sums)
        for row in sums.to_dict('records'):
            di, ri = dates.index(row['date']), REGIMES.index(row['regime'])
            counts[di, pi, ri] = row['n']
            for mi, method in enumerate(METHODS):
                cube[di, pi, ri, mi] = [row[method+'_'+c+'_sse'] for c in CHANNELS]
        audits.append(dict(panel=panel, n=len(frame), nodes=len(links), isolated_nodes=len(isolated_links),
            prediction_source_sha256=sha(path), topology_sha256=sha(topology_path)))
        print(panel, 'targets', len(frame), 'union fallback', masks['union'].sum(axis=0).tolist(), flush=True)

    n, total = counts.sum(axis=0), cube.sum(axis=0)
    rmse, state = score_sse(total, n)
    metric_rows = []
    for pi, panel in enumerate(panels):
        for mi, method in enumerate(METHODS):
            for ri, regime in enumerate(REGIMES):
                metric_rows.append(dict(panel=panel, method=method, regime=regime, n=int(n[pi, ri]),
                    rmse_speed_kmh=float(rmse[pi, ri, mi, 0]),
                    rmse_flow_per_lane=float(rmse[pi, ri, mi, 1]), S_state=float(state[pi, ri, mi])))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT/'regime_metrics.csv', index=False)
    overview = metrics.groupby('method', sort=False)[['S_state', 'rmse_speed_kmh', 'rmse_flow_per_lane']].mean().reset_index()
    references = pd.read_csv(SOURCE/'method_summary.csv').set_index('method')
    for method in ['LightGBM', 'Transformer']:
        assert abs(overview.set_index('method').loc[method, 'S_state']-references.loc[method, 'S_state']) < 1e-12
    overview['delta_vs_Transformer'] = overview.S_state - references.loc['Transformer', 'S_state']
    overview['delta_vs_LightGBM'] = overview.S_state - references.loc['LightGBM', 'S_state']
    overview.to_csv(OUT/'method_summary.csv', index=False)
    comparison = metrics.groupby(['panel', 'method']).S_state.mean().unstack('method')[METHODS]
    for rule in RULES:
        np.testing.assert_allclose(comparison[rule+'_both'] - comparison.Transformer,
            comparison[rule+'_speed_only'] + comparison[rule+'_flow_only'] - 2*comparison.Transformer, rtol=0, atol=1e-12)
    comparison['union_delta_vs_Transformer'] = comparison.union_both-comparison.Transformer
    comparison['union_delta_vs_LightGBM'] = comparison.union_both-comparison.LightGBM
    comparison.to_csv(OUT/'comparison.csv')
    pd.DataFrame(route_rows).to_csv(OUT/'routing_counts.csv', index=False)
    group_frame = pd.DataFrame(group_rows)
    group_frame['rmse'] = np.sqrt(group_frame.sse / group_frame.n.replace(0, np.nan))
    group_frame.to_csv(OUT/'group_metrics_by_corridor.csv', index=False)
    pooled = group_frame.groupby(['channel', 'group', 'method'], as_index=False)[['n', 'sse']].sum()
    pooled['rmse'] = np.sqrt(pooled.sse/pooled.n.replace(0, np.nan))
    pooled.to_csv(OUT/'pooled_group_metrics.csv', index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(OUT/'daily_errors.csv', index=False)

    rng = np.random.default_rng(2026)
    samples = []
    while len(samples) < 1000:
        pick = rng.integers(len(dates), size=len(dates))
        sampled_n = counts[pick].sum(axis=0)
        if (sampled_n > 0).all():
            _, score = score_sse(cube[pick].sum(axis=0), sampled_n)
            samples.append(score.mean(axis=(0, 1)))
    samples = np.array(samples)
    observed = state.mean(axis=(0, 1))
    intervals = []
    for mi, method in enumerate(METHODS[2:], start=2):
        for baseline in ['Transformer', 'LightGBM']:
            bi = METHODS.index(baseline)
            delta = samples[:, mi]-samples[:, bi]
            intervals.append(dict(method=method, baseline=baseline, observed_delta=float(observed[mi]-observed[bi]),
                ci_low=float(np.quantile(delta, .025)), ci_high=float(np.quantile(delta, .975)), replicates=1000))
    pd.DataFrame(intervals).to_csv(OUT/'paired_calendar_bootstrap.csv', index=False)
    write_json('checks.json', dict(training_started=False, inference_started=False, total_internal_targets=int(n.sum()),
        exact_routing='passed for each channel and every target', baselines_reproduced_tolerance=1e-12,
        channel_additivity_tolerance=1e-12, topology_counts_match_prior_audit=True,
        bootstrap='1000 paired shared-calendar-day resamples across all ten corridors, seed 2026; conditional on frozen selected models',
        inputs=audits))
    write_json('status.json', dict(state='complete', completed=panels))
    print('\nMETHODS\n'+overview.to_string(index=False), flush=True)
    print('\nCORRIDORS\n'+comparison[['LightGBM', 'Transformer', 'union_both', 'union_delta_vs_Transformer']].to_string(), flush=True)


if __name__ == '__main__':
    main()
