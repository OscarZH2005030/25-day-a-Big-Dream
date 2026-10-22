"""Exact direct-regression labels, attached to existing cached training sample IDs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .cache import snapshot
from .cache_builder import IDENTIFIER_COLUMNS
from .data import JOIN_KEY, KEY, TARGETS, day_files, eligible, network, read_day, sha256

DIRECT_TARGET_COLUMNS = ['__target_speed', '__target_flow_lane']


def aligned_direct_targets(training, labels):
    if not training[IDENTIFIER_COLUMNS].reset_index(drop=True).equals(
            labels[IDENTIFIER_COLUMNS].reset_index(drop=True)):
        raise ValueError('Direct-label row IDs/order do not match the feature cache')
    result = labels[DIRECT_TARGET_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite direct labels')
    return result


def exact_day_targets(keys, complete, net):
    """Use released training truth, never residual + rounded float32 initial values."""
    matched = keys[JOIN_KEY].merge(complete[JOIN_KEY + TARGETS + ['pct_observed']],
                                  on=JOIN_KEY, how='left', validate='one_to_one')
    if not eligible(matched).all():
        raise ValueError('Direct-label sample has missing/unavailable training truth')
    lanes = keys.link_id.map(net.lanes).to_numpy(dtype=float)
    if not np.isfinite(lanes).all() or (lanes <= 0).any():
        raise ValueError('Invalid lane count in direct labels')
    labels = matched[TARGETS].to_numpy(dtype=np.float64, copy=True)
    labels[:, 1] /= lanes
    return labels


def direct_label_cache(cache, root, panel, training_folder, training_manifest):
    dates = training_manifest['metadata']['fit_dates']
    full_files = day_files(root, panel, 'train', False)
    net_path = root / 'corridors' / panel / 'network'
    inputs = [full_files[d] for d in dates] + [net_path / 'links.csv', net_path / 'fd_parameters.csv']
    identity = dict(kind='direct_labels', panel=panel, schema='exact_training_truth_v1',
        parent_training_key=training_manifest['key'],
        sample_table_sha256=next(a['sha256'] for a in training_manifest['artifacts']
                                 if a['name'] == 'training.parquet'),
        target_units=['km/h', 'veh/h/lane'], builder_sha256=sha256(Path(__file__)), inputs=snapshot(inputs))

    def build(folder):
        sample = pd.read_parquet(training_folder / 'training.parquet', columns=IDENTIFIER_COLUMNS)
        keys = sample.rename(columns={c: c[2:] for c in IDENTIFIER_COLUMNS}).copy()
        if not keys.panel.eq(panel).all() or keys.duplicated(KEY).any():
            raise ValueError('Invalid cached training sample IDs')
        keys['__row'] = np.arange(len(keys))
        labels = np.full((len(keys), 2), np.nan, dtype=np.float64)
        net = network(root, panel)
        grouped = keys.groupby(keys.timestamp.str[:10], sort=True)
        if sorted(grouped.groups) != dates:
            raise ValueError('Cached training sample dates do not match fit dates')
        for i, (date, group) in enumerate(grouped, 1):
            labels[group['__row'].to_numpy(dtype=int)] = exact_day_targets(
                group, read_day(full_files[date]), net)
            if i % 25 == 0 or i == len(dates):
                print(f'[{panel}] exact direct labels {i}/{len(dates)} days', flush=True)
        result = sample.copy()
        result[DIRECT_TARGET_COLUMNS] = labels
        aligned_direct_targets(sample, result)
        result.to_parquet(folder / 'direct_targets.parquet', index=False)
        return dict(training_rows=len(sample), fit_dates=dates, exact_truth=True,
                    label_columns=DIRECT_TARGET_COLUMNS, evaluation_truth_used=False)

    return cache.get_or_build(identity, build)
