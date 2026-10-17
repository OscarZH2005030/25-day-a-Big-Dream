"""Input-only Task 1 target filtering; no labels, predictions or training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REFERENCE = Path(__file__).resolve().parents[1]/'full_dataset_training_v2/runs/all_corridors_time80_20'


def classify_day(folder, names):
    keys = pd.read_parquet(folder/'keys.parquet')
    ii = np.load(folder/'ii.npy'); jj = np.load(folder/'jj.npy')
    features = np.load(folder/'features.npy', mmap_mode='r')[jj, ii]
    visible = np.load(folder/'visible.npy', mmap_mode='r').astype(bool)
    def feature(name):
        return features[:, names.index(name)].astype('float64')
    result = keys.copy()
    relative = feature('relative_speed_initial')
    result['relative_speed_initial'] = relative.astype('float32')
    result['congestion_hint'] = np.select([~np.isfinite(relative), relative < .5, relative < .8],
                                         ['unknown', 'low_speed', 'transition'], default='near_free_flow')
    result['capacity_utilization_initial'] = feature('capacity_utilization_initial').astype('float32')
    ticks = np.broadcast_to(np.arange(288)[:, None], visible.shape[:2])
    for channel, prefix in enumerate(['speed_kmh', 'flow_vph']):
        label = 'speed' if channel == 0 else 'flow'
        seen = visible[:, :, channel]
        if seen[jj, ii].any():
            raise ValueError('Template targets unexpectedly visible: ' + str(folder))
        left = np.maximum.accumulate(np.where(seen, ticks, -1), axis=0) + 1
        right = np.minimum.accumulate(np.where(seen, ticks, 288)[::-1], axis=0)[::-1] - 1
        length = right[jj, ii] - left[jj, ii] + 1
        both = np.isfinite(feature(prefix+'_prev_gap_min')) & np.isfinite(feature(prefix+'_next_gap_min'))
        neighbors = feature(prefix+'_up_count') + feature(prefix+'_down_count')
        spatial = neighbors > 0
        prev, nxt = feature(prefix+'_prev'), feature(prefix+'_next')
        scale = feature('free_speed_kmh') if channel == 0 else feature('capacity_vph')/feature('lanes')
        delta = np.abs(nxt-prev)/np.maximum(scale, 1e-6)
        gap = np.select([length == 288, ~both, length <= 3, length <= 12],
                        ['whole_day', 'day_edge', 'short_5_15min', 'medium_20_60min'], default='long_over60min')
        # Candidate strategies only; thresholds are descriptive, not validated routing rules.
        candidate = np.select([
            both & (length <= 3) & (delta <= .1),
            both & (delta > .2) & spatial,
            spatial,
            both], ['interpolation_candidate', 'graph_change_candidate', 'graph_candidate', 'temporal_candidate'],
            default='history_fallback_candidate')
        result[label+'_gap_minutes_in_day'] = (length*5).astype('int16')
        result[label+'_gap_group'] = gap
        result[label+'_both_sides'] = both
        result[label+'_neighbor_count'] = neighbors.astype('int16')
        result[label+'_endpoint_change_ratio'] = delta.astype('float32')
        result[label+'_candidate'] = candidate
    if result.duplicated(['panel','timestamp','station_id','link_id','mask_regime']).any():
        raise ValueError('Duplicate template keys')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'holdout_profile')
    p.add_argument('--phase', choices=['training','internal','holdout','all_cached'], default='holdout')
    p.add_argument('--panel', help='One corridor; default all ten')
    p.add_argument('--regime', choices=['R1','R2','R3'])
    p.add_argument('--channel', choices=['speed','flow'], default='speed')
    p.add_argument('--gap-group', choices=['whole_day','day_edge','short_5_15min','medium_20_60min','long_over60min'])
    p.add_argument('--congestion', choices=['unknown','low_speed','transition','near_free_flow'])
    p.add_argument('--neighbor', choices=['available','none'])
    p.add_argument('--summary-only', action='store_true')
    args = p.parse_args()
    plan = json.loads((args.reference/'plan.json').read_text())
    phase_dates = {'training':plan['training_dates'], 'internal':plan['internal_validation_dates'], 'holdout':plan['holdout_dates']}
    dates = sum(phase_dates.values(), []) if args.phase == 'all_cached' else phase_dates[args.phase]
    phase_for = {d:k for k, values in phase_dates.items() for d in values}
    panels = [args.panel] if args.panel else plan['panels']
    if any(panel not in plan['panels'] for panel in panels):
        p.error('Unknown corridor')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'COMPLETE.json').exists():
        p.error('Output already complete; choose a different --output for a new filter')
    sys.path.insert(0, str(args.reference.parent.parent))
    from training_io import RunLock
    with RunLock(args.output/'filter.lock'):
        manifest = args.output/'filter_config.json'
        config = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise ValueError('Output belongs to a different filter; choose a new --output')
        manifest.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        summaries, locations, daily = [], [], []
        selected = original = 0
        for panel in panels:
            out = args.output/panel; out.mkdir(exist_ok=True)
            if not args.summary_only and (out/'targets.parquet').exists():
                (out/'targets.parquet').unlink()
            names = json.loads((args.reference/panel/'cache/ready.json').read_text())['names']
            writer = None
            try:
                for index, date in enumerate(dates):
                    frame = classify_day(args.reference/panel/'cache'/date, names)
                    frame['phase'] = phase_for[date]
                    original += len(frame)
                    keep = np.ones(len(frame), dtype=bool)
                    if args.regime: keep &= frame.mask_regime == args.regime
                    if args.gap_group: keep &= frame[args.channel+'_gap_group'] == args.gap_group
                    if args.congestion: keep &= frame.congestion_hint == args.congestion
                    if args.neighbor: keep &= (frame[args.channel+'_neighbor_count'] > 0) == (args.neighbor == 'available')
                    frame = frame.loc[keep].copy(); selected += len(frame)
                    if len(frame):
                        if not args.summary_only:
                            table = pa.Table.from_pandas(frame, preserve_index=False)
                            if writer is None: writer = pq.ParquetWriter(out/'targets.parquet', table.schema, compression='zstd')
                            writer.write_table(table)
                        for channel in ['speed','flow']:
                            group = ['panel','phase','mask_regime',channel+'_gap_group','congestion_hint',channel+'_candidate']
                            a = frame.groupby(group, dropna=False).size().rename('n').reset_index()
                            a = a.rename(columns={channel+'_gap_group':'gap_group',channel+'_candidate':'candidate'})
                            a['channel']=channel; summaries.append(a)
                        station_frame = frame[['panel','station_id','link_id']].copy()
                        station_frame['n'] = 1
                        station_frame['speed_long'] = (frame.speed_gap_minutes_in_day>60).astype('int32')
                        station_frame['speed_no_neighbors'] = (frame.speed_neighbor_count==0).astype('int32')
                        locations.append(station_frame.groupby(['panel','station_id','link_id'],as_index=False)[['n','speed_long','speed_no_neighbors']].sum())
                        daily.append(dict(panel=panel,date=date,phase=phase_for[date],n=len(frame)))
                    if index % 10 == 0 or index+1 == len(dates):
                        print(f'{panel}: {index+1}/{len(dates)} dates; selected total {selected:,}',flush=True)
            finally:
                if writer: writer.close()
        if summaries:
            cols=['panel','phase','mask_regime','channel','gap_group','congestion_hint','candidate']
            pd.concat(summaries).groupby(cols,dropna=False,as_index=False)['n'].sum().to_csv(args.output/'group_counts.csv',index=False,encoding='utf-8-sig')
            where = pd.concat(locations).groupby(['panel','station_id','link_id'],as_index=False)[['n','speed_long','speed_no_neighbors']].sum()
            where['selected_target_fraction_of_slots'] = where.n/(len(dates)*288)
            where.sort_values('n',ascending=False).to_csv(args.output/'locations.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(daily).to_csv(args.output/'dates.csv',index=False,encoding='utf-8-sig')
        record=dict(original_template_targets=original,selected_template_targets=selected,phase=args.phase,
                    filters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    labels_or_predictions_read=False,
                    limitations=['Day-edge gap lengths are truncated at midnight; prev/next search is within day.',
                                 'Initial speed and endpoint change are input-based hints, not true congestion labels.',
                                 'Candidate strategies are hypotheses; no optimal predictor has been selected.',
                                 'Scope: cached official train dates from 2030-06-29; not official validation/private.'])
        (args.output/'COMPLETE.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'DONE: {selected:,}/{original:,} template targets; '+str(args.output),flush=True)


if __name__ == '__main__': main()
