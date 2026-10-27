"""Validate per-task queue outputs without requiring withheld official labels."""
import numpy as np
import pandas as pd

KEYS = ['window_id', 'timestamp', 'link_id']


def validate_queue(frame, template):
    if not set(KEYS + ['queue_pred']).issubset(frame):
        raise ValueError('Required queue columns missing')
    a, b = frame.copy(), template.copy()
    for x in (a, b):
        for col in ['window_id', 'link_id']:
            if x[col].isna().any():
                raise ValueError('Null queue identifier')
            x[col] = x[col].astype(str)
        x['timestamp'] = pd.to_datetime(x.timestamp, utc=True, errors='raise')
        if x.timestamp.isna().any() or x.duplicated(KEYS).any():
            raise ValueError('Missing timestamps or duplicated queue keys')
    values = pd.to_numeric(a.queue_pred, errors='raise')
    if not np.isfinite(values).all() or not values.isin([0, 1]).all():
        raise ValueError('queue_pred must be finite binary 0/1')
    merged = b[KEYS].merge(a[KEYS + ['queue_pred']], on=KEYS, how='outer',
                           indicator=True, validate='one_to_one')
    if len(a) != len(b) or not merged['_merge'].eq('both').all():
        raise ValueError('Queue rows must exactly match the template')
    return {'passed': True, 'rows': len(a), 'binary': True,
            'official_queue_score': None, 'scope': 'schema_only_no_hidden_truth'}
