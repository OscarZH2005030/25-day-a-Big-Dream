from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch

from ..wave_neural.data import Windows


class ContextWindows(Windows):
    """Only the 13 wave feature channels change; the readout prior never changes."""

    @lru_cache(maxsize=8)
    def day(self, date):
        day = Windows.day.__wrapped__(self, date)
        with np.load(Path(self.records[date]['folder'])/'inputs.npz', allow_pickle=False) as archive:
            day['prior'] = np.maximum(archive['interp'], 0).astype(np.float32)
        if not np.isfinite(day['prior']).all():
            raise ValueError('Invalid common temporal prior')
        return day

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        batch['window_index'] = torch.tensor(index, dtype=torch.int64)
        return batch


class Coverage:
    """Unique supervised target rows visited by all-node temporal windows."""

    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.counts = {date: np.zeros(288, dtype=np.int64) for date in dataset.dates}
        for date, rows in keys.groupby(keys.timestamp.str[:10]):
            if date not in self.counts:
                raise ValueError('Training key is outside dataset dates')
            times = pd.to_datetime(rows.timestamp, utc=True)
            slots = (times.dt.hour*12+times.dt.minute//5).to_numpy()
            self.counts[date] = np.bincount(slots, minlength=288)
        if sum(int(v.sum()) for v in self.counts.values()) != len(keys):
            raise ValueError('Target coverage scope differs')
        self.seen = {date: np.zeros(288, dtype=bool) for date in dataset.dates}
        self.windows = set()

    def update(self, indices):
        for index in indices:
            index = int(index)
            date, start = self.dataset.items[index]
            self.seen[date][start:start+self.dataset.window] = True
            self.windows.add(index)

    def summary(self):
        return dict(unique_targets_seen=sum(int(self.counts[d][self.seen[d]].sum()) for d in self.seen),
                    unique_windows_seen=len(self.windows))


def order_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def anchor_check(records_train, records_eval, source, means, attributes):
    """Check actual first/center/last days in both splits before any training."""
    checks = []
    for split, records in [('train', records_train), ('eval', records_eval)]:
        args = (records, source['grid_'+split]['metadata'], means, attributes)
        off = ContextWindows(*args, False, split=='train')
        on = ContextWindows(*args, True, split=='train')
        try:
            for date in sorted({off.dates[0], off.dates[len(off.dates)//2], off.dates[-1]}):
                a, b = off.day(date), on.day(date)
                if not np.array_equal(a['prior'], b['prior']) or not np.array_equal(a['features'][..., :18], b['features'][..., :18]):
                    raise ValueError('Common interpolation prior or public channels differ')
                if np.any(a['features'][..., 18:]):
                    raise ValueError('Wave-off channels must be zero')
                for name in ['x', 'mask', 'y', 'supervise']:
                    if not np.array_equal(a[name], b[name]):
                        raise ValueError('Paired dataset differs: '+name)
                checks.append(dict(split=split, date=date, same_prior=True, same_common_channels=True,
                                   wave_off_zero=True, wave_on_nonzero=bool(np.any(b['features'][..., 18:]))))
        finally:
            off.day.cache_clear(); on.day.cache_clear()
    return dict(passed=True, prior='same cached temporal interpolation for both models at every output',
                common_channels=18, optional_wave_channels=13, checked_days=checks)
