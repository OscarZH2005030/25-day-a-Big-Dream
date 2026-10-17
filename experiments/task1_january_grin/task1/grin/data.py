"""Daily graph tensors; fit-only normalization and exact baseline-style samples."""
from __future__ import annotations

from functools import lru_cache
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..cache import snapshot
from ..data import KEY, TARGETS, day_files, network, read_csv, read_day, template
from ..features import multihop_neighbors


def select_samples(group, rng, count):
    if len(group) > count:
        group = group.iloc[np.sort(rng.choice(len(group), count, replace=False))]
    return group.reset_index(drop=True)


def window_starts(window, stride, steps=288):
    if not 1 <= stride <= window <= steps:
        raise ValueError('Require 1 <= stride <= window <= 288')
    return sorted(set(range(0, steps - window + 1, stride)) | {steps - window})


def node_table(frame, net):
    nodes = frame[['station_id', 'link_id']].drop_duplicates().sort_values('station_id').reset_index(drop=True)
    if nodes.station_id.duplicated().any() or nodes.link_id.duplicated().any():
        raise ValueError('Expected one unique mainline link per station')
    nodes['lanes'] = nodes.link_id.map(net.lanes)
    if not np.isfinite(nodes.lanes).all() or (nodes.lanes <= 0).any():
        raise ValueError('Invalid node lane counts')
    return nodes


def graph_adjacency(nodes, net, topology):
    # A[receiver,sender]: downstream station receives from its nearest upstream
    # instrumented mainline station, bridging unsensed links. Reverse support is
    # separately normalized by the model. No connector/ramp nodes or self loops.
    links = nodes.link_id.tolist(); positions = {link: i for i, link in enumerate(links)}
    neighbors = multihop_neighbors(topology, set(links), net, max_hops=1)
    adjacency = np.zeros((len(links), len(links)), np.float32)
    for origin in links:
        for destination, distance in neighbors.get((origin, 'down', 1), []):
            adjacency[positions[destination], positions[origin]] = 1 / max(float(distance), .01)
    if not np.isfinite(adjacency).all() or not adjacency.any():
        raise ValueError('No valid graph edges')
    return adjacency


def day_grid(frame, nodes):
    pairs = frame[['station_id', 'link_id']].drop_duplicates().sort_values('station_id').reset_index(drop=True)
    if not pairs.equals(nodes[['station_id', 'link_id']].reset_index(drop=True)):
        raise ValueError('Station/link mapping changed between days')
    if frame.duplicated(['slot', 'station_id']).any() or frame.date.nunique() != 1:
        raise ValueError('Expected one full daily station grid')
    index = pd.MultiIndex.from_product([range(288), nodes.station_id], names=['slot', 'station_id'])
    grid = frame.set_index(['slot', 'station_id']).reindex(index)
    if grid.link_id.isna().any():
        raise ValueError('Incomplete five-minute daily grid')
    raw = grid[TARGETS].to_numpy(dtype=float, copy=True).reshape(288, len(nodes), 2)
    raw[:, :, 1] /= nodes.lanes.to_numpy()[None, :]
    quality = pd.to_numeric(grid.pct_observed, errors='coerce').to_numpy().reshape(288, len(nodes), 1) >= 75
    mask = np.isfinite(raw) & quality
    return np.where(mask, raw, 0).astype(np.float32), mask, raw


def key_positions(keys, nodes):
    station_index = pd.Series(np.arange(len(nodes)), index=nodes.station_id)
    positions = keys.station_id.map(station_index)
    if positions.isna().any():
        raise ValueError('Unknown target station')
    positions = positions.to_numpy(int)
    if not np.array_equal(nodes.link_id.to_numpy()[positions], keys.link_id.to_numpy()):
        raise ValueError('Target station/link mismatch')
    ts = pd.to_datetime(keys.timestamp, utc=True)
    return (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy(int), positions


def grid_cache(cache, root, cfg, mode, parent=None):
    if mode not in ['training', 'evaluation']:
        raise ValueError('Unknown grid mode')
    start, end = (cfg['fit_start'], cfg['fit_end']) if mode == 'training' else (cfg['eval_start'], cfg['eval_end'])
    dates = pd.date_range(start, end).strftime('%Y-%m-%d').tolist()
    limit = cfg.get('max_fit_days' if mode == 'training' else 'max_eval_days')
    if limit:
        dates = dates[:limit]
    masked = day_files(root, cfg['panel'], 'train', True)
    full = day_files(root, cfg['panel'], 'train', False) if mode == 'training' else {}
    if set(dates) - set(masked) or (mode == 'training' and set(dates) - set(full)):
        raise ValueError('Required daily files are missing')
    net = network(root, cfg['panel'])
    if parent is None and mode != 'training':
        raise ValueError('Evaluation requires the training node/scaler specification')
    nodes = node_table(read_day(masked[dates[0]]), net) if parent is None else pd.DataFrame(parent['metadata']['nodes'])
    folder = root / 'corridors' / cfg['panel'] / 'network'
    inputs = [masked[d] for d in dates] + [folder / name for name in ['links.csv', 'fd_parameters.csv', 'lwr_mainline_topology.csv']]
    inputs += [root / 'task1' / cfg['panel'] / 'train/sample_submission_state.csv', root / 'config/competition_contract.json']
    if mode == 'training':
        inputs += [full[d] for d in dates]
    identity = dict(kind='grin_' + mode, panel=cfg['panel'], schema='grin_daily_grid_v1', dates=dates,
        release_root=str(root.resolve()), parent_training_key=parent['key'] if parent else None,
        sample_per_day=cfg['sample_per_day'] if mode == 'training' else None,
        seed=cfg['seed'] if mode == 'training' else None, inputs=snapshot(inputs),
        implementation=snapshot([Path(__file__), Path(__file__).parents[1] / 'data.py', Path(__file__).parents[1] / 'features.py']),
        packages={p: importlib.metadata.version(p) for p in ['numpy','pandas','pyarrow']},
        evaluation_truth_used=False)

    def build(destination):
        all_targets = template(root, cfg['panel'], 'train')
        all_targets = all_targets.loc[all_targets.timestamp.str[:10].isin(dates)].reset_index(drop=True)
        if sorted(all_targets.timestamp.str[:10].unique()) != dates:
            raise ValueError('Template and grid dates differ')
        all_targets.to_parquet(destination / 'expected_keys.parquet', index=False)
        topology = read_csv(folder / 'lwr_mainline_topology.csv')
        adjacency = graph_adjacency(nodes, net, topology)
        np.save(destination / 'adjacency.npy', adjacency)
        rng = np.random.default_rng(cfg['seed'])
        sums, squares, counts = np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64)
        sample_parts = []
        for i, date in enumerate(dates, 1):
            frame = read_day(masked[date])
            x, mask, raw = day_grid(frame, nodes)
            targets = all_targets.loc[all_targets.timestamp.str.startswith(date)]
            rr, cc = key_positions(targets, nodes)
            if np.isfinite(raw[rr, cc]).any():
                raise ValueError('A competition target is visible in the masked grid')
            arrays = dict(x=x, mask=mask)
            if mode == 'training':
                selected = select_samples(targets, rng, cfg['sample_per_day'])
                sample_parts.append(selected)
                rr, cc = key_positions(selected, nodes)
                _, valid_truth, truth = day_grid(read_day(full[date]), nodes)
                if not valid_truth[rr, cc].all():
                    raise ValueError('Selected target truth is unavailable or ineligible')
                y, supervise = np.zeros_like(x), np.zeros_like(mask)
                y[rr, cc] = truth[rr, cc]; supervise[rr, cc] = True
                arrays.update(y=y, supervise=supervise)
                for ch in range(2):
                    values = x[:, :, ch][mask[:, :, ch]].astype(np.float64)
                    sums[ch] += values.sum(); squares[ch] += np.square(values).sum(); counts[ch] += len(values)
            np.savez_compressed(destination / (date + '.npz'), **arrays)
            if i % 10 == 0 or i == len(dates):
                print(f'[GRIN {mode} cache] {i}/{len(dates)} days', flush=True)
        if mode == 'training':
            if (counts == 0).any():
                raise ValueError('No observed training values for normalization')
            mean = sums / counts
            std = np.sqrt(np.maximum(squares / counts - mean ** 2, 1.))
            scaler = dict(mean=mean.tolist(), std=std.tolist(), fit_dates=dates,
                          fit_on='Only eligible visible masked training observations')
            samples = pd.concat(sample_parts, ignore_index=True)
            samples.to_parquet(destination / 'training_sample_keys.parquet', index=False)
            sampled_rows = len(samples)
        else:
            scaler = parent['metadata']['scaler']; sampled_rows = None
        return dict(nodes=nodes.to_dict('records'), dates=dates, scaler=scaler,
            target_rows=len(all_targets), sampled_training_rows=sampled_rows,
            graph_edges=int((adjacency > 0).sum()), isolated_nodes=int(((adjacency.sum(0) + adjacency.sum(1)) == 0).sum()),
            mode=mode, evaluation_truth_used=False, channels=['speed_kmh', 'flow_vph_per_lane'])

    return cache.get_or_build(identity, build)


def time_covariates(date, nodes):
    angle = np.arange(288) * (2 * np.pi / 288)
    weekday = pd.Timestamp(date).dayofweek * (2 * np.pi / 7)
    features = np.stack([np.sin(angle), np.cos(angle), np.full(288, np.sin(weekday)),
                         np.full(288, np.cos(weekday))], -1).astype(np.float32)
    return np.broadcast_to(features[:, None, :], (288, nodes, 4)).copy()


class GraphWindows(Dataset):
    def __init__(self, folder, manifest, window=48, stride=24):
        self.folder, self.metadata = Path(folder), manifest['metadata']
        self.window = window
        self.items = [(date, s) for date in self.metadata['dates'] for s in window_starts(window, stride)]
        self.mean = np.asarray(self.metadata['scaler']['mean'], np.float32)
        self.std = np.asarray(self.metadata['scaler']['std'], np.float32)

    @lru_cache(maxsize=16)
    def day(self, date):
        with np.load(self.folder / (date + '.npz'), allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        date, start = self.items[index]; stop = start + self.window
        day = self.day(date); mask = day['mask'][start:stop]
        x = np.where(mask, (day['x'][start:stop] - self.mean) / self.std, 0).astype(np.float32)
        if 'y' not in day:
            raise ValueError('Evaluation grids must not be used as a training dataset')
        supervise = day['supervise'][start:stop]
        y = np.where(supervise, (day['y'][start:stop] - self.mean) / self.std, 0).astype(np.float32)
        u = time_covariates(date, x.shape[1])[start:stop]
        return tuple(torch.from_numpy(a.copy()) for a in [x, mask, u, y, supervise])
