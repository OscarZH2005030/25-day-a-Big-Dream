from __future__ import annotations

import numpy as np
import pandas as pd

from .data import KEY, TARGETS


def validate_predictions(expected: pd.DataFrame, predicted: pd.DataFrame):
    if predicted.duplicated(KEY).any():
        raise ValueError('Duplicate prediction keys')
    coverage = expected[KEY].merge(predicted[KEY], on=KEY, how='outer', indicator=True)
    if not coverage['_merge'].eq('both').all():
        raise ValueError(f'Target coverage mismatch: {coverage._merge.value_counts().to_dict()}')
    if not np.isfinite(predicted[TARGETS].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite predictions')
    if (predicted[TARGETS] < 0).any().any():
        raise ValueError('Negative predictions')


def score(pred: pd.DataFrame, truth: pd.DataFrame, net: pd.DataFrame,
          family: str, method: str) -> tuple[list[dict], pd.DataFrame]:
    validate_predictions(truth, pred)
    data = truth[KEY + TARGETS].merge(pred, on=KEY, suffixes=('_true', '_pred'),
                                     validate='one_to_one')
    lanes = data.link_id.map(net.lanes).to_numpy(dtype=float)
    data['speed_sq_error'] = (data.speed_kmh_pred - data.speed_kmh_true) ** 2
    data['flow_lane_sq_error'] = ((data.flow_vph_pred - data.flow_vph_true) / lanes) ** 2
    data['date'] = data.timestamp.str[:10]
    free = data.link_id.map(net.free_speed_kmh)
    data['congested'] = data.speed_kmh_true <= 0.60 * free
    rows = []
    for regime, g in data.groupby('mask_regime'):
        rv = float(np.sqrt(g.speed_sq_error.mean()))
        rq = float(np.sqrt(g.flow_lane_sq_error.mean()))
        sv, sq = max(0., 1 - rv / 25), max(0., 1 - rq / 600)
        rows.append(dict(method=method, panel=g.panel.iloc[0], family=family,
                         regime=regime, n=len(g), rmse_speed_kmh=rv,
                         rmse_flow_per_lane=rq, S_speed=sv, S_flow=sq,
                         S_state=0.54 * sv + 0.46 * sq))
    return rows, data


def aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    """Regime -> direction -> corridor family -> overall, never row-weighted."""
    panel = detail.groupby(['method', 'family', 'panel'], as_index=False).S_state.mean()
    panel['regimes_present'] = panel.apply(
        lambda r: detail.loc[(detail.method == r.method) & (detail.panel == r.panel),
                             'regime'].nunique(), axis=1)
    family = panel.groupby(['method', 'family'], as_index=False).S_state.mean()
    overall = family.groupby('method', as_index=False).S_state.mean()
    overall['panel_count'] = panel.groupby('method').size().reindex(overall.method).to_numpy()
    overall['family_count'] = family.groupby('method').size().reindex(overall.method).to_numpy()
    overall['all_regimes_present'] = panel.groupby('method').regimes_present.min().reindex(
        overall.method).to_numpy() == 3
    return overall
