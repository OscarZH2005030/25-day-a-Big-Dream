"""Chronological Task 1 pilot: history, interpolation, direct and residual LGBM.

All input features come from masked observations. Unmasked observations are read
only for pre-training historical profiles and supervised targets. The default
profile/train/holdout blocks are disjoint. This is an offline reconstruction
experiment, not a causal Task 2 forecast or an official validation submission.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

CHANNELS = ['speed_kmh', 'flow_vph']
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']


def log(message):
    print(message, flush=True)


def score(truth, pred, regimes, method):
    rows = []
    for regime in ['R1', 'R2', 'R3']:
        keep = regimes == regime
        if not keep.any():
            raise ValueError(f'Holdout has no {regime} targets')
        rmse = np.sqrt(np.mean((pred[keep] - truth[keep]) ** 2, axis=0))
        sv, sq = max(0., 1-rmse[0]/25), max(0., 1-rmse[1]/600)
        rows.append(dict(method=method, regime=regime, n=int(keep.sum()),
                         rmse_speed_kmh=float(rmse[0]),
                         rmse_flow_per_lane=float(rmse[1]),
                         S_speed=sv, S_flow=sq, S_state=.54*sv+.46*sq))
    rows.append(dict(method=method, regime='macro_mean', n=len(truth),
                     S_state=float(np.mean([x['S_state'] for x in rows]))))
    return rows


class Experiment:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.release_root)
        self.panel = self.root/'corridors'/args.panel
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(args.seed)
        sample = pd.read_parquet(sorted((self.panel/'train/mainline_states_masked').rglob('*.parquet'))[0])
        stations = sample[['station_id', 'link_id']].drop_duplicates().sort_values('station_id')
        if stations.station_id.duplicated().any():
            raise ValueError('A station maps to multiple links')
        self.stations = stations.station_id.tolist()
        self.links = stations.link_id.tolist()
        self.si = {s: i for i, s in enumerate(self.stations)}
        self.n = len(self.stations)
        fd = pd.read_csv(self.panel/'network/fd_parameters.csv').set_index('link_id')
        self.static = fd.reindex(self.links)
        if self.static[['lanes', 'free_speed_kmh', 'length_km', 'capacity_vph']].isna().any().any():
            raise ValueError('Incomplete road metadata')
        self.lanes = self.static.lanes.to_numpy()
        topology = pd.read_csv(self.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
        self.neighbors = {}
        for side, col in [('up', 'incoming_link_ids'), ('down', 'outgoing_link_ids')]:
            self.neighbors[side] = []
            for link in self.links:
                ids = str(topology.loc[link, col]).split(';') if link in topology.index else []
                self.neighbors[side].append([i for i, other in enumerate(self.links) if other in ids])
        self.truth_files = {p.stem[-10:].replace('_', '-'): p for p in (self.panel/'train/mainline_states').rglob('*.parquet')}
        self.masked_files = {p.stem[-10:].replace('_', '-'): p for p in (self.panel/'train/mainline_states_masked').rglob('*.parquet')}

    def grid(self, frame, col):
        ts = pd.to_datetime(frame.timestamp, utc=True)
        slots = (ts.dt.hour*12 + ts.dt.minute//5).to_numpy()
        ids = frame.station_id.map(self.si).to_numpy()
        if pd.isna(ids).any() or frame.duplicated(['station_id', 'timestamp']).any():
            raise ValueError('Unknown or duplicate station-time key')
        result = np.full((self.n, 288), np.nan)
        result[ids.astype(int), slots] = pd.to_numeric(frame[col], errors='coerce')
        return result

    def profile(self):
        sums = np.zeros((2, self.n, 7, 288))
        counts = np.zeros_like(sums)
        dates = sorted(d for d in self.truth_files if d < self.args.train_start)
        if not dates:
            raise ValueError('No historical profile dates')
        for j, date in enumerate(dates):
            x = pd.read_parquet(self.truth_files[date])
            wd = pd.Timestamp(date).dayofweek
            for c, col in enumerate(CHANNELS):
                a = self.grid(x, col)
                if c == 1:
                    a = a/self.lanes[:, None]
                valid = np.isfinite(a) & (self.grid(x, 'pct_observed') >= 75)
                sums[c, :, wd] += np.where(valid, a, 0)
                counts[c, :, wd] += valid
            if (j+1) % 50 == 0:
                log(f'Historical profile: {j+1}/{len(dates)} days')
        fallback = sums.sum(axis=(2, 3))/np.maximum(counts.sum(axis=(2, 3)), 1)
        self.history = np.divide(sums, counts, out=np.broadcast_to(fallback[:, :, None, None], sums.shape).copy(), where=counts>0)
        np.savez_compressed(self.out/'history.npz', mean=self.history,
                            stations=np.array(self.stations), links=np.array(self.links))
        self.profile_dates = dates

    def targets(self):
        # Read the template, not the set of nulls. Keep only model/holdout dates.
        parts = []
        path = self.root/'task1'/self.args.panel/'train/sample_submission_state.csv'
        for chunk in pd.read_csv(path, usecols=KEYS, dtype=str, chunksize=250000):
            dates = chunk.timestamp.str[:10]
            keep = (dates >= self.args.train_start) & (dates < self.args.val_end)
            parts.append(chunk.loc[keep])
        selected = pd.concat(parts, ignore_index=True)
        if selected.duplicated(KEYS).any():
            raise ValueError('Duplicate target keys')
        self.day_targets = {d: g.reset_index(drop=True) for d, g in selected.groupby(selected.timestamp.str[:10], sort=True)}

    def feature_grid(self, date):
        x = pd.read_parquet(self.masked_files[date])
        wd = pd.Timestamp(date).dayofweek
        hist = self.history[:, :, wd]
        features = {}
        base = []
        ticks = np.broadcast_to(np.arange(288), (self.n, 288))
        features['weekday'] = np.full(ticks.shape, wd)
        features['slot_sin'] = np.sin(2*np.pi*ticks/288)
        features['slot_cos'] = np.cos(2*np.pi*ticks/288)
        for col in ['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph']:
            features[col] = np.broadcast_to(self.static[col].to_numpy()[:, None], ticks.shape)
        for c, col in enumerate(CHANNELS):
            a = self.grid(x, col)
            if c == 1:
                a = a/self.lanes[:, None]
            valid = np.isfinite(a)
            prev = np.maximum.accumulate(np.where(valid, ticks, -1), axis=1)
            nxt = np.minimum.accumulate(np.where(valid, ticks, 288)[:, ::-1], axis=1)[:, ::-1]
            pv = np.take_along_axis(a, np.clip(prev, 0, 287), axis=1)
            nv = np.take_along_axis(a, np.clip(nxt, 0, 287), axis=1)
            pv[prev<0] = np.nan
            nv[nxt>287] = np.nan
            both = (prev>=0) & (nxt<288)
            fraction = (ticks-prev)/np.maximum(nxt-prev, 1)
            # Outside the observed span use history; avoid propagating a distant
            # endpoint indefinitely. Within the span use linear interpolation.
            initial = np.where(both, pv+(nv-pv)*fraction, hist[c])
            base.append(initial)
            features[col+'_history'] = hist[c]
            features[col+'_initial'] = initial
            features[col+'_prev'] = pv
            features[col+'_next'] = nv
            features[col+'_prev_gap_min'] = np.where(prev>=0, (ticks-prev)*5, np.nan)
            features[col+'_next_gap_min'] = np.where(nxt<288, (nxt-ticks)*5, np.nan)
            features[col+'_history_fallback'] = (~both).astype(float)
            for offset in [-6, -3, -1, 1, 3, 6]:
                b = np.full_like(a, np.nan)
                if offset>0:
                    b[:, :-offset] = a[:, offset:]
                else:
                    b[:, -offset:] = a[:, :offset]
                features[f'{col}_offset_{offset*5}min'] = b
            for side, groups in self.neighbors.items():
                neighbor = np.full_like(a, np.nan)
                available = np.zeros_like(a)
                for i, ids in enumerate(groups):
                    if ids:
                        z = a[ids]
                        count = np.isfinite(z).sum(axis=0)
                        neighbor[i] = np.divide(np.nansum(z, axis=0), count,
                                                out=np.full(288, np.nan), where=count>0)
                        available[i] = count
                features[f'{col}_{side}'] = neighbor
                features[f'{col}_{side}_count'] = available
        for col in ['occupancy', 'pct_observed', 'is_imputed']:
            a = self.grid(x, col)
            # Contemporaneous target occupancy is blank; shifted values provide
            # usable surrounding evidence. Quality flags themselves are released.
            if col != 'occupancy':
                features[col] = a
            for offset in [-1, 1]:
                b = np.full_like(a, np.nan)
                if offset>0:
                    b[:, :-1] = a[:, 1:]
                else:
                    b[:, 1:] = a[:, :-1]
                features[f'{col}_offset_{offset*5}min'] = b
        features['relative_speed_initial'] = base[0]/self.static.free_speed_kmh.to_numpy()[:, None]
        features['capacity_utilization_initial'] = base[1]*self.lanes[:, None]/self.static.capacity_vph.to_numpy()[:, None]
        return features, base, hist, x

    def features(self, date, target):
        features, base, hist, x = self.feature_grid(date)
        ts = pd.to_datetime(target.timestamp, utc=True)
        ii = target.station_id.map(self.si).to_numpy(dtype=int)
        jj = (ts.dt.hour*12+ts.dt.minute//5).to_numpy()
        if np.isfinite(self.grid(x, 'speed_kmh')[ii, jj]).any() or np.isfinite(self.grid(x, 'flow_vph')[ii, jj]).any():
            raise ValueError('Target values remain visible in masked inputs')
        if not np.array_equal(target.link_id.to_numpy(), np.array(self.links)[ii]):
            raise ValueError('Template station-link mismatch')
        X = np.column_stack([v[ii, jj] for v in features.values()]).astype('float32')
        if np.isinf(X).any():
            raise ValueError('Infinite features')
        initial = np.column_stack([v[ii, jj] for v in base])
        historical = np.column_stack([v[ii, jj] for v in hist])
        # Read answers only after construction of all model inputs.
        truth = pd.read_parquet(self.truth_files[date])
        y = np.column_stack([self.grid(truth, c)[ii, jj] for c in CHANNELS])
        y[:, 1] /= self.lanes[ii]
        if not np.isfinite(y).all() or not np.isfinite(initial).all():
            raise ValueError('Non-finite target or initial value')
        return X, y, initial, historical, self.lanes[ii], list(features)

    def collect(self, validation):
        arrays = [[] for _ in range(6)]
        keys = []
        for date, full_target in self.day_targets.items():
            is_val = self.args.val_start <= date < self.args.val_end
            is_train = self.args.train_start <= date < self.args.val_start
            if (validation and not is_val) or (not validation and not is_train):
                continue
            target = full_target
            if not validation and len(target)>self.args.samples_per_day:
                ids = np.sort(self.rng.choice(len(target), self.args.samples_per_day, replace=False))
                target = target.iloc[ids].reset_index(drop=True)
            X, y, base, hist, lanes, self.names = self.features(date, target)
            for items, a in zip(arrays, [X, y, base, hist, lanes, target.mask_regime.to_numpy()]):
                items.append(a)
            keys.append(target)
            log(f'{"Holdout" if validation else "Train"} {date}: {len(target):,} targets')
        if not keys:
            raise ValueError('Empty chronological split')
        return [np.concatenate(a) for a in arrays], pd.concat(keys, ignore_index=True)

    def run(self):
        started = time.time()
        self.profile()
        self.targets()
        train, _ = self.collect(False)
        valid, keys = self.collect(True)
        X, y, base, _, _, _ = train
        V, vy, vb, vh, lanes, regimes = valid
        rows = score(vy, np.maximum(vh, 0), regimes, 'historical')
        rows += score(vy, np.maximum(vb, 0), regimes, 'interpolation')
        params = dict(n_estimators=self.args.trees, learning_rate=.05, num_leaves=31,
                      min_child_samples=80, reg_lambda=1., colsample_bytree=.9,
                      n_jobs=self.args.threads, random_state=self.args.seed,
                      verbosity=-1, deterministic=True, force_col_wise=True)
        importance = []
        for mode in ['direct', 'residual']:
            prediction = np.zeros_like(vy)
            for c, channel in enumerate(CHANNELS):
                log(f'Fit {mode}: {channel}, {len(y):,} training rows, {len(self.names)} features')
                model = LGBMRegressor(**params)
                label = y[:, c] - base[:, c] if mode=='residual' else y[:, c]
                model.fit(X, label, feature_name=self.names)
                prediction[:, c] = model.booster_.predict(V)+(vb[:, c] if mode=='residual' else 0)
                model.booster_.save_model(str(self.out/f'{mode}_{channel}.txt'))
                for name, gain in zip(self.names, model.booster_.feature_importance(importance_type='gain')):
                    importance.append(dict(model=mode, target=channel, feature=name, gain=float(gain)))
            prediction = np.maximum(prediction, 0)
            rows += score(vy, prediction, regimes, mode)
            output = keys.copy()
            output['speed_kmh'] = prediction[:, 0]
            output['flow_vph'] = prediction[:, 1]*lanes
            output.to_parquet(self.out/f'{mode}_holdout_predictions.parquet', index=False)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(self.out/'metrics.csv', index=False)
        pd.DataFrame(importance).to_csv(self.out/'feature_importance.csv', index=False)
        manifest = dict(vars(self.args), historical_start=self.profile_dates[0],
                        historical_end=self.profile_dates[-1], train_rows=len(y),
                        holdout_rows=len(vy), features=self.names, model_parameters=params,
                        elapsed_seconds=time.time()-started,
                        limitations=['Single corridor, one held-out month; not a competition total score.',
                                     'Within-day interpolation; no ramp features or physics constraints yet.',
                                     'Historical profile is frozen before the model-training block.',
                                     'All four methods use the same complete held-out template target set.'])
        (self.out/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
        log(metrics[metrics.regime=='macro_mean'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-root', default='/Users/liuchengwei/kaggle_public')
    parser.add_argument('--panel', default='D12_I5_N')
    parser.add_argument('--train-start', default='2031-01-01')
    parser.add_argument('--val-start', default='2031-02-01')
    parser.add_argument('--val-end', default='2031-03-01', help='Exclusive date')
    parser.add_argument('--samples-per-day', type=int, default=10000)
    parser.add_argument('--trees', type=int, default=200)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', default=str(Path(__file__).parent/'task1_pilot'))
    args = parser.parse_args()
    if not args.train_start < args.val_start < args.val_end:
        parser.error('Require train-start < val-start < val-end')
    Experiment(args).run()
