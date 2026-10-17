"""Evaluate one frozen, topology-only routing rule on existing holdout predictions.

No model imports, training, model inference, thresholds, or rule selection.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
TF = Path(__file__).resolve().parents[1]/'full_dataset_training_v2/runs/all_corridors_time80_20'
LGB = HERE.parent/'lightgbm_time80_20/runs/all_corridors_time80_20'
INTERNAL = HERE.parent/'internal_gap_switch/results'
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
METHODS = ['LightGBM', 'Transformer', 'isolated_both', 'isolated_speed_only', 'isolated_flow_only']
REGIMES = ['R1', 'R2', 'R3']
CHANNELS = ['speed', 'flow_per_lane']
SCALE = np.array([25., 600.])


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(name, value):
    tmp = (OUT/name).with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(OUT/name)


def evaluate(sse, n):
    rmse = np.sqrt(sse/n[..., None, None])
    component = np.maximum(0., 1-rmse/SCALE)
    return rmse, component @ np.array([.54, .46])


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise SystemExit('Already complete. Preserve the frozen results.')
    plan = json.loads((TF/'plan.json').read_text())
    other_plan = json.loads((LGB/'plan.json').read_text())
    assert Path(other_plan['reference_run']).resolve() == TF.resolve()
    assert other_plan['plan_sha256'] == sha(TF/'plan.json')
    for key in ['history_dates', 'training_dates', 'internal_validation_dates', 'holdout_dates']:
        assert plan[key] == other_plan[key], key
    dates, panels = plan['holdout_dates'], plan['panels']
    assert dates == list(pd.date_range('2031-01-05', '2031-02-28').strftime('%Y-%m-%d'))
    # Freeze the protocol on disk before reading holdout labels or scoring.
    write_json('plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        dates=dates, panels=panels, default='Transformer', fallback='LightGBM',
        primary_rule='no non-self neighbor in original model graph => LightGBM, else Transformer',
        original_graph='released modeled nodes; incoming/outgoing edges symmetrized, 1/2-hop plus self',
        isolation='zero non-self degree in original graph; equivalent for 1-hop and 2-hop',
        temporal_endpoint_condition=False, training=False, inference=False, tuning=False,
        methods=METHODS, script_sha256=sha(Path(__file__)), transformer_run=str(TF), lightgbm_run=str(LGB),
        aggregation='equal mean of R1/R2/R3 within corridor, then equal mean of ten corridors',
        flow_unit='veh/h/lane for errors; routed parquet flow_vph converted back to total',
        caveat='Previously inspected holdout, not an untouched test; not an official competition score.'))
    n_cube = np.zeros((len(dates), len(panels), 3), dtype=np.int64)
    sse_cube = np.zeros((*n_cube.shape, len(METHODS), 2))
    reference_topology = pd.read_csv(HERE.parents[2]/'docs/results/task1_time80_20/neighbor_failure_analysis/topology_audit.csv').set_index('panel')
    route_rows, group_rows, audit_rows, daily_rows = [], [], [], []
    max_reference_difference = 0.
    for pi, panel in enumerate(panels):
        folder = OUT/panel
        folder.mkdir(exist_ok=True)
        for run in [TF, LGB]:
            assert (run/panel/'COMPLETE.json').exists()
        tp = TF/panel/'holdout_predictions.parquet'
        lp = LGB/panel/'holdout_predictions.parquet'
        a, b = pd.read_parquet(tp), pd.read_parquet(lp)
        assert not a.duplicated(KEYS).any() and not b.duplicated(KEYS).any()
        assert a[KEYS].astype(str).equals(b[KEYS].astype(str))
        assert set(a.panel) == {panel}
        assert set(a.timestamp.astype(str).str[:10]) == set(dates)
        frozen = json.loads((INTERNAL/panel/'COMPLETE.json').read_text())
        model_hashes = [sha(TF/panel/'best.pt')] + [sha(LGB/panel/(c+'.txt')) for c in ['speed_kmh', 'flow_per_lane']]
        assert model_hashes == [frozen['checkpoint_sha256']] + frozen['lgb_model_sha256']
        with np.load(TF/panel/'history.npz', allow_pickle=False) as history:
            links, stations = history['links'].tolist(), history['stations'].tolist()
        linkset = set(links)
        topology_path = Path(plan['release_root'])/'corridors'/panel/'network/lwr_mainline_topology.csv'
        top = pd.read_csv(topology_path).set_index('mainline_link_id')
        assert len(links) == len(linkset) and linkset == set(top.index)
        neighbors = {link: set() for link in links}
        for link in links:
            row = top.loc[link]
            for other in str(row.incoming_link_ids).split(';')+str(row.outgoing_link_ids).split(';'):
                if other in linkset and other != link:
                    neighbors[link].add(other)
                    neighbors[other].add(link)
        isolated_links = {link for link, ns in neighbors.items() if not ns}
        assert len(links) == reference_topology.loc[panel, 'nodes']
        assert len(isolated_links) == reference_topology.loc[panel, 'isolated']
        isolated = a.link_id.isin(isolated_links).to_numpy()
        fd = pd.read_csv(topology_path.parent/'fd_parameters.csv').set_index('link_id')
        lanes = a.link_id.map(fd.lanes).to_numpy(dtype='float64')
        assert np.isfinite(lanes).all() and (lanes > 0).all()
        tf = np.column_stack([a.speed_kmh.to_numpy(), a.flow_vph.to_numpy()/lanes])
        lg = np.column_stack([b.speed_kmh.to_numpy(), b.flow_vph.to_numpy()/lanes])
        hybrid = np.where(isolated[:, None], lg, tf)
        predictions = [lg, tf, hybrid, np.column_stack([hybrid[:, 0], tf[:, 1]]), np.column_stack([tf[:, 0], hybrid[:, 1]])]
        assert all(np.isfinite(p).all() and (p >= 0).all() for p in predictions)
        for ci in range(2):
            np.testing.assert_array_equal(hybrid[:, ci], np.where(isolated, lg[:, ci], tf[:, ci]))
        selected = a[KEYS].copy()
        selected['speed_kmh'] = hybrid[:, 0]
        # Select the original total flow directly to avoid a needless divide/multiply round trip.
        selected['flow_vph'] = np.where(isolated, b.flow_vph, a.flow_vph)
        selected['flow_per_lane'] = hybrid[:, 1]
        selected['use_lightgbm'] = isolated
        selected.to_parquet(folder/'holdout_routed_predictions.parquet', index=False)
        cursor, truth_chunks = 0, []
        for di, date in enumerate(dates):
            cache = TF/panel/'cache'/date
            keys = pd.read_parquet(cache/'keys.parquet')
            ii, jj = np.load(cache/'ii.npy'), np.load(cache/'jj.npy')
            end = cursor+len(ii)
            assert len(keys) == len(ii) == len(jj)
            assert a[KEYS].iloc[cursor:end].reset_index(drop=True).astype(str).equals(keys[KEYS].astype(str))
            np.testing.assert_array_equal(keys.link_id.to_numpy(), np.array(links)[ii])
            np.testing.assert_array_equal(keys.station_id.to_numpy(), np.array(stations)[ii])
            assert not np.load(cache/'visible.npy', mmap_mode='r')[jj, ii].astype(bool).any()
            truth = np.load(cache/'label.npy', mmap_mode='r')[jj, ii].astype('float64')*SCALE
            assert np.isfinite(truth).all()
            truth_chunks.append(truth)
            values = pd.DataFrame({'regime': keys.mask_regime.to_numpy(), 'n': 1})
            for mi, method in enumerate(METHODS):
                error = predictions[mi][cursor:end]-truth
                for ci, channel in enumerate(CHANNELS):
                    values[method+'_'+channel+'_sse'] = error[:, ci]**2
            sums = values.groupby('regime', as_index=False).sum()
            sums['date'], sums['panel'] = date, panel
            daily_rows.append(sums)
            for row in sums.to_dict('records'):
                ri = REGIMES.index(row['regime'])
                n_cube[di, pi, ri] = row['n']
                for mi, method in enumerate(METHODS):
                    sse_cube[di, pi, ri, mi] = [row[method+'_'+c+'_sse'] for c in CHANNELS]
            cursor = end
        assert cursor == len(a)
        truth = np.concatenate(truth_chunks)
        for mi, method in enumerate(['LightGBM', 'Transformer']):
            run = LGB if mi == 0 else TF
            reference = pd.read_csv(run/panel/'metrics.csv').set_index('regime')
            for ri, regime in enumerate(REGIMES):
                keep = a.mask_regime.to_numpy() == regime
                rmse = np.sqrt(np.mean((predictions[mi][keep]-truth[keep])**2, axis=0))
                expected = reference.loc[regime, ['rmse_speed_kmh', 'rmse_flow_per_lane']].to_numpy(dtype=float)
                difference = float(np.max(np.abs(rmse-expected)))
                assert difference < 1e-7, (panel, method, regime, difference)
                max_reference_difference = max(max_reference_difference, difference)
        for group, mask in [('isolated', isolated), ('connected', ~isolated)]:
            for mi, method in enumerate(METHODS[:3]):
                for ci, channel in enumerate(CHANNELS):
                    error = predictions[mi][mask, ci]-truth[mask, ci]
                    group_rows.append(dict(panel=panel, group=group, method=method, channel=channel,
                        n=int(mask.sum()), sse=float(np.sum(error**2))))
        route_rows.append(dict(panel=panel, nodes=len(links), isolated_nodes=len(isolated_links),
            n=len(a), fallback_n=int(isolated.sum()), fallback_fraction=float(isolated.mean())))
        audits = dict(panel=panel, targets=len(a), transformer_prediction_sha256=sha(tp),
            lightgbm_prediction_sha256=sha(lp), topology_sha256=sha(topology_path),
            model_sha256=model_hashes, frozen_models_match_internal=True)
        audit_rows.append(audits)
        write_json(panel+'/COMPLETE.json', audits)
        print(panel, 'checked targets', len(a), 'fallback', int(isolated.sum()), flush=True)

    n, sse = n_cube.sum(axis=0), sse_cube.sum(axis=0)
    rmse, state = evaluate(sse, n)
    rows = []
    for pi, panel in enumerate(panels):
        for ri, regime in enumerate(REGIMES):
            for mi, method in enumerate(METHODS):
                rows.append(dict(panel=panel, regime=regime, method=method, n=int(n[pi, ri]),
                    rmse_speed_kmh=float(rmse[pi, ri, mi, 0]), rmse_flow_per_lane=float(rmse[pi, ri, mi, 1]),
                    S_speed=float(max(0, 1-rmse[pi, ri, mi, 0]/25)),
                    S_flow=float(max(0, 1-rmse[pi, ri, mi, 1]/600)), S_state=float(state[pi, ri, mi])))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT/'regime_metrics.csv', index=False)
    macro = metrics.groupby(['panel', 'method'], sort=False)[['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_speed', 'S_flow', 'S_state']].mean().reset_index()
    macro.to_csv(OUT/'corridor_metrics.csv', index=False)
    comparison = macro.pivot(index='panel', columns='method', values=['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state'])
    comparison.columns = ['__'.join(c) for c in comparison.columns]
    comparison = comparison.reindex(panels)
    for field in ['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_state']:
        comparison[field+'__hybrid_minus_tf'] = comparison[field+'__isolated_both']-comparison[field+'__Transformer']
    comparison.to_csv(OUT/'comparison.csv')
    for pi, panel in enumerate(panels):
        np.testing.assert_allclose(state[pi, :, 2]-state[pi, :, 1],
            state[pi, :, 3]+state[pi, :, 4]-2*state[pi, :, 1], rtol=0, atol=1e-12)
    summary = macro.groupby('method', sort=False)[['rmse_speed_kmh', 'rmse_flow_per_lane', 'S_speed', 'S_flow', 'S_state']].mean()
    summary['delta_vs_Transformer'] = summary.S_state-summary.loc['Transformer', 'S_state']
    summary['delta_vs_LightGBM'] = summary.S_state-summary.loc['LightGBM', 'S_state']
    summary.to_csv(OUT/'method_summary.csv')
    for run, method in [(TF, 'Transformer'), (LGB, 'LightGBM')]:
        reference = json.loads((run/'summary.json').read_text())
        score_reference = reference.get('mean_corridor_S_state')
        if score_reference is not None:
            assert abs(summary.loc[method, 'S_state']-score_reference) < 1e-7
    pd.DataFrame(route_rows).to_csv(OUT/'routing_counts.csv', index=False)
    groups = pd.DataFrame(group_rows)
    groups['rmse'] = np.sqrt(groups.sse/groups.n.replace(0, np.nan))
    groups.to_csv(OUT/'structure_group_metrics.csv', index=False)
    pooled = groups.groupby(['group', 'channel', 'method'], as_index=False)[['n', 'sse']].sum()
    pooled['rmse'] = np.sqrt(pooled.sse/pooled.n)
    pooled.to_csv(OUT/'pooled_structure_group_metrics.csv', index=False)
    pd.concat(daily_rows, ignore_index=True).to_csv(OUT/'daily_errors.csv', index=False)
    np.savez_compressed(OUT/'daily_error_cube.npz', n=n_cube, sse=sse_cube,
        methods=np.array(METHODS), panels=np.array(panels), dates=np.array(dates))

    # Shared date draws preserve same-day correlations between regimes and corridors.
    rng = np.random.default_rng(2026)
    sampled_scores, sampled_rmse = [], []
    while len(sampled_scores) < 1000:
        pick = rng.integers(len(dates), size=len(dates))
        sampled_n = n_cube[pick].sum(axis=0)
        if (sampled_n > 0).all():
            error, score = evaluate(sse_cube[pick].sum(axis=0), sampled_n)
            sampled_scores.append(score.mean(axis=1))
            sampled_rmse.append(error.mean(axis=1))
    sampled_scores, sampled_rmse = np.array(sampled_scores), np.array(sampled_rmse)
    intervals = []
    for baseline in ['Transformer', 'LightGBM']:
        bi = METHODS.index(baseline)
        for pi, panel in list(enumerate(panels))+[(None, 'equal_corridor_mean')]:
            for field in ['S_state', 'rmse_speed_kmh', 'rmse_flow_per_lane']:
                if field == 'S_state':
                    values = sampled_scores[:, :, 2]-sampled_scores[:, :, bi]
                    observed = state[:, :, 2].mean(axis=1)-state[:, :, bi].mean(axis=1)
                else:
                    ci = 0 if field == 'rmse_speed_kmh' else 1
                    values = sampled_rmse[:, :, 2, ci]-sampled_rmse[:, :, bi, ci]
                    observed = rmse[:, :, 2, ci].mean(axis=1)-rmse[:, :, bi, ci].mean(axis=1)
                delta = values.mean(axis=1) if pi is None else values[:, pi]
                intervals.append(dict(panel=panel, baseline=baseline, field=field,
                    observed_delta=float(observed.mean() if pi is None else observed[pi]),
                    ci_low=float(np.quantile(delta, .025)), ci_high=float(np.quantile(delta, .975)), replicates=1000))
    pd.DataFrame(intervals).to_csv(OUT/'paired_calendar_bootstrap.csv', index=False)
    check = dict(training_started=False, inference_started=False, temporal_condition_applied=False,
        total_holdout_targets=int(n.sum()), fallback_targets=sum(r['fallback_n'] for r in route_rows),
        prediction_keys='exactly matched between models and all 550 cache key files; no duplicates',
        hidden_targets='all target channel values invisible to input', exact_channel_routing='passed',
        original_model_graph='reconstructed from saved model-node order and official topology',
        frozen_models='all ten checkpoint and twenty LightGBM hashes match internal evaluation',
        baseline_rmse_max_difference=max_reference_difference, baseline_rmse_tolerance=1e-7,
        channel_score_additivity='passed within 1e-12',
        bootstrap='1000 shared-calendar-day paired resamples; fixed models, seed 2026; descriptive',
        inputs=audit_rows)
    write_json('checks.json', check)
    write_json('COMPLETE.json', dict(state='complete', completed=panels, total_targets=int(n.sum()),
        S_state=float(summary.loc['isolated_both', 'S_state']), finished_utc=datetime.now(timezone.utc).isoformat()))
    print('\nSUMMARY\n'+summary.to_string(), flush=True)
    print('\nCOMPARISON\n'+comparison.to_string(), flush=True)


if __name__ == '__main__':
    main()
