from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
JOIN_KEY = ['timestamp', 'station_id', 'link_id']
TARGETS = ['speed_kmh', 'flow_vph']
ID_TYPES = {'station_id': str, 'link_id': str, 'panel': str, 'mask_regime': str}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=ID_TYPES)


def template(root: Path, panel: str, split: str) -> pd.DataFrame:
    result = pd.read_csv(root / 'task1' / panel / split / 'sample_submission_state.csv',
                         usecols=KEY, dtype=ID_TYPES)
    if result.duplicated(KEY).any():
        raise ValueError(f'{panel}/{split}: duplicate template keys')
    if result[KEY].isna().any().any():
        raise ValueError(f'{panel}/{split}: null template keys')
    return result


def day_files(root: Path, panel: str, split: str, masked: bool = True) -> dict[str, Path]:
    layer = 'mainline_states_masked' if masked else 'mainline_states'
    result = {}
    for path in sorted((root / 'corridors' / panel / split / layer).rglob('*.parquet')):
        # Daily release partitions use synthetic_mainline_YYYY_MM_DD.parquet.
        parts = path.stem.split('_')[-3:]
        date = '-'.join(parts)
        if len(date) != 10 or not date[:4].isdigit():
            raise ValueError(f'Unrecognized daily filename: {path.name}')
        if date in result:
            raise ValueError(f'Duplicate day {date}: {result[date]}, {path}')
        result[date] = path
    if not result:
        raise FileNotFoundError(f'No daily parquet files: {panel}/{split}/{layer}')
    return result


def read_day(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index(drop=True)
    for name in ['station_id', 'link_id']:
        frame[name] = frame[name].astype(str)
    if frame.duplicated(JOIN_KEY).any():
        raise ValueError(f'Duplicate observation keys in {path}')
    ts = pd.to_datetime(frame.timestamp, utc=True)
    frame['slot'] = ts.dt.hour * 12 + ts.dt.minute // 5
    frame['weekday'] = ts.dt.dayofweek
    frame['date'] = ts.dt.strftime('%Y-%m-%d')
    if not ((ts.dt.minute % 5 == 0) & (ts.dt.second == 0)).all():
        raise ValueError(f'Non five-minute timestamps: {path}')
    return frame


def eligible(frame: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(frame.pct_observed, errors='coerce').ge(75)
            & np.isfinite(frame[TARGETS]).all(axis=1))


def network(root: Path, panel: str) -> pd.DataFrame:
    folder = root / 'corridors' / panel / 'network'
    links = read_csv(folder / 'links.csv').set_index('link_id')
    fd = read_csv(folder / 'fd_parameters.csv')
    if fd.link_id.duplicated().any():
        # This release is per link. Fail explicitly if a different release is used.
        raise ValueError('Expected one FD row per link; inspect release schema')
    fd = fd.set_index('link_id')
    for col in ['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph']:
        if col in fd:
            links[col] = fd[col].combine_first(links.get(col))
    if links.index.duplicated().any():
        raise ValueError('Duplicate network link IDs')
    return links


class HistoricalProfile:
    """Link x weekday/time profile, with exact leave-day-out sufficient statistics."""

    def __init__(self, links: list[str]):
        self.links = links
        self.index = {link: i for i, link in enumerate(links)}
        self.total = np.zeros((len(links), 2016, 2), dtype=np.float64)
        self.count = np.zeros_like(self.total)
        self.fit_dates: list[str] = []

    def accumulate(self, frame: pd.DataFrame, sign: float = 1):
        f = frame.loc[eligible(frame)]
        li = f.link_id.map(self.index)
        if li.isna().any():
            raise ValueError('Unknown link in profile')
        slot = (f.weekday * 288 + f.slot).to_numpy(dtype=int)
        idx = li.to_numpy(dtype=int)
        for c, col in enumerate(TARGETS):
            np.add.at(self.total[:, :, c], (idx, slot), sign * f[col].to_numpy())
            np.add.at(self.count[:, :, c], (idx, slot), sign)

    def lookup(self, frame: pd.DataFrame) -> np.ndarray:
        li = frame.link_id.map(self.index)
        if li.isna().any():
            raise ValueError('Unknown link at inference')
        idx = li.to_numpy(dtype=int)
        slots = (frame.weekday * 288 + frame.slot).to_numpy(dtype=int)
        counts, sums = self.count[idx, slots], self.total[idx, slots]
        link_count = self.count.sum(axis=1)
        link_mean = np.divide(self.total.sum(axis=1), link_count,
                              out=np.full_like(link_count, np.nan), where=link_count > 0)
        global_count = self.count.sum(axis=(0, 1))
        global_mean = np.divide(self.total.sum(axis=(0, 1)), global_count,
                                out=np.full(2, np.nan), where=global_count > 0)
        fallback = np.where(np.isfinite(link_mean[idx]), link_mean[idx], global_mean)
        answer = np.divide(sums, counts, out=fallback.copy(), where=counts > 0)
        if not np.isfinite(answer).all():
            raise ValueError('Insufficient profile history. Use more training days.')
        return answer


def fit_profile(root: Path, panel: str, start: str, end: str) -> HistoricalProfile:
    files = day_files(root, panel, 'train', False)
    selected = [(d, p) for d, p in files.items() if start <= d <= end]
    if not selected:
        raise ValueError('No dates selected for profile')
    all_links = network(root, panel).index.tolist()
    profile = HistoricalProfile(all_links)
    for date, path in selected:
        profile.accumulate(read_day(path))
        profile.fit_dates.append(date)
    return profile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
