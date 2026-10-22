from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import feature_implementation, snapshot
from .data import JOIN_KEY, KEY, TARGETS, day_files, fit_profile, network, read_csv, read_day
from .features import build_features
from .pipeline import assert_hidden, select_rows, selected_targets

RESIDUAL_COLUMNS = ['__residual_speed', '__residual_flow_lane']
IDENTIFIER_COLUMNS = ['__' + c for c in KEY]


def network_inputs(root, panel):
    n = root / 'corridors' / panel / 'network'
    return [n / name for name in ['links.csv', 'fd_parameters.csv', 'lwr_mainline_topology.csv']] + [
        root / 'config' / 'competition_contract.json']


def dates_in_range(root, panel, split, start, end, limit):
    dates = [d for d in sorted(day_files(root, panel, split)) if start <= d <= end]
    if limit:
        dates = dates[:limit]
    if not dates:
        raise ValueError('No dates selected for feature cache')
    return dates


def training_cache(cache, root: Path, panel: str, cfg):
    dates = dates_in_range(root, panel, 'train', cfg['fit_start'], cfg['fit_end'], cfg.get('max_fit_days'))
    masked, full = day_files(root, panel, 'train'), day_files(root, panel, 'train', False)
    inputs = network_inputs(root, panel) + [root / 'task1' / panel / 'train' / 'sample_submission_state.csv']
    inputs += [masked[d] for d in dates] + [full[d] for d in dates]
    print(f'[{panel}] fingerprinting training inputs', flush=True)
    identity = dict(kind='train', panel=panel, release_root=str(root.resolve()), dates=dates,
                    sample_per_day=cfg['sample_per_day'], seed=cfg['seed'],
                    max_gap_minutes=cfg['max_gap_minutes'], profile='leave_entire_day_out',
                    implementation=feature_implementation(), inputs=snapshot(inputs))

    def build(folder):
        targets = selected_targets(root, panel, 'train', dates[0], dates[-1])
        if sorted(targets.timestamp.str[:10].unique()) != dates:
            raise ValueError('Template dates differ from cache date list')
        net = network(root, panel)
        topology = read_csv(root / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv')
        profile = fit_profile(root, panel, dates[0], dates[-1])
        rng, chunks = np.random.default_rng(cfg['seed']), []
        columns = None
        for i, (date, group) in enumerate(targets.groupby(targets.timestamp.str[:10], sort=True), 1):
            masked_day, full_day = read_day(masked[date]), read_day(full[date])
            positions = select_rows(masked_day, group)
            assert_hidden(masked_day, positions)
            if len(positions) > cfg['sample_per_day']:
                take = np.sort(rng.choice(len(group), cfg['sample_per_day'], replace=False))
                group, positions = group.iloc[take], positions[take]
            profile.accumulate(full_day, -1)
            try:
                x, initial, _, lanes = build_features(masked_day, profile, net, topology,
                                                       cfg['max_gap_minutes'], feature_set='all')
            finally:
                profile.accumulate(full_day, 1)
            labels = group[JOIN_KEY].merge(full_day[JOIN_KEY + TARGETS], on=JOIN_KEY,
                       how='left', validate='one_to_one')[TARGETS].to_numpy(dtype=float, copy=True)
            labels[:, 1] /= lanes[positions]
            residual = labels - initial[positions]
            if not np.isfinite(residual).all():
                raise ValueError('Non-finite cache labels')
            chunk = x.iloc[positions].reset_index(drop=True).copy()
            if columns is None:
                columns = list(chunk.columns)
            if list(chunk.columns) != columns:
                raise ValueError('Feature schema changed between days')
            chunk[RESIDUAL_COLUMNS] = residual
            chunk[IDENTIFIER_COLUMNS] = group[KEY].to_numpy()
            chunks.append(chunk)
            if i % 5 == 0 or i == len(dates):
                print(f'[{panel}] cache training {i}/{len(dates)} days', flush=True)
        table = pd.concat(chunks, ignore_index=True)
        table.to_parquet(folder / 'training.parquet', index=False)
        with (folder / 'profile.pkl').open('wb') as stream:
            pickle.dump(profile, stream)
        return dict(feature_columns=columns, training_rows=len(table), fit_dates=dates,
                    sample_key_policy='same ordered RNG sampling as original B2')

    return cache.get_or_build(identity, build)


def evaluation_cache(cache, root, panel, cfg, training_folder, training_manifest):
    dates = dates_in_range(root, panel, 'train', cfg['eval_start'], cfg['eval_end'], cfg.get('max_eval_days'))
    if set(dates).intersection(training_manifest['metadata']['fit_dates']):
        raise ValueError('Fit and evaluation dates overlap')
    masked = day_files(root, panel, 'train')
    inputs = network_inputs(root, panel) + [root / 'task1' / panel / 'train' / 'sample_submission_state.csv']
    inputs += [masked[d] for d in dates]
    print(f'[{panel}] fingerprinting evaluation inputs', flush=True)
    identity = dict(kind='evaluation', panel=panel, release_root=str(root.resolve()), dates=dates,
                    training_key=training_manifest['key'], max_gap_minutes=cfg['max_gap_minutes'],
                    profile_sha256=next(a['sha256'] for a in training_manifest['artifacts'] if a['name'] == 'profile.pkl'),
                    implementation=feature_implementation(), inputs=snapshot(inputs), labels_in_cache=False)

    def build(folder):
        targets = selected_targets(root, panel, 'train', dates[0], dates[-1])
        with (training_folder / 'profile.pkl').open('rb') as stream:
            profile = pickle.load(stream)
        net = network(root, panel)
        topology = read_csv(root / 'corridors' / panel / 'network' / 'lwr_mainline_topology.csv')
        files, total = [], 0
        for i, (date, group) in enumerate(targets.groupby(targets.timestamp.str[:10], sort=True), 1):
            f = read_day(masked[date])
            pos = select_rows(f, group)
            assert_hidden(f, pos)
            x, initial, history, lanes = build_features(f, profile, net, topology,
                                  cfg['max_gap_minutes'], feature_set='all')
            chunk = x.iloc[pos].reset_index(drop=True).copy()
            chunk[IDENTIFIER_COLUMNS] = group[KEY].to_numpy()
            chunk[['__initial_speed', '__initial_flow_lane']] = initial[pos]
            chunk[['__history_speed', '__history_flow_lane']] = history[pos]
            chunk['__lanes'] = lanes[pos]
            name = date + '.parquet'
            chunk.to_parquet(folder / name, index=False)
            files.append(name)
            total += len(chunk)
            if i % 5 == 0 or i == len(dates):
                print(f'[{panel}] cache evaluation {i}/{len(dates)} days', flush=True)
        if sorted(targets.timestamp.str[:10].unique()) != dates:
            raise ValueError('Evaluation template dates differ from cache date list')
        return dict(feature_columns=list(x.columns), eval_dates=dates, files=files, target_rows=total)

    return cache.get_or_build(identity, build)
