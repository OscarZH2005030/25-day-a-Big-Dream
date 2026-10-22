from __future__ import annotations

import numpy as np
import pandas as pd

from task1.metrics import aggregate
from task1.transformer_priority.common import metric_rows

NEURAL = {'T0': 0.0, 'T01_equal': 0.5, 'T1': 1.0}
WEIGHTS = (0.0, 0.1, 0.25, 0.5)


def mix(t0, t1, tree, isolated, beta, speed=0.0, flow=0.0):
    """All arrays use published physical units; preserve exact endpoint values."""
    arrays = [np.asarray(a, dtype=np.float64) for a in (t0, t1, tree)]
    if any(a.shape != arrays[0].shape for a in arrays) or arrays[0].ndim != 2 or arrays[0].shape[1] != 2:
        raise ValueError('Prediction shapes differ')
    if any(not np.isfinite(a).all() or (a < 0).any() for a in arrays):
        raise ValueError('Invalid predictions')
    if any(w < 0 or w > 1 or not np.isfinite(w) for w in (beta, speed, flow)):
        raise ValueError('Invalid fusion weight')
    isolated = np.asarray(isolated)
    if isolated.dtype != bool or isolated.shape != (len(arrays[0]),):
        raise ValueError('Isolation mask mismatch')
    a, b, tree = arrays
    neural = a.copy() if beta == 0 else b.copy() if beta == 1 else (1-beta)*a + beta*b
    result = neural.copy()
    for c, alpha in enumerate((speed, flow)):
        if alpha:
            result[:, c] = tree[:, c] if alpha == 1 else (1-alpha)*neural[:, c] + alpha*tree[:, c]
        if alpha == 0 and not np.array_equal(result[~isolated, c], neural[~isolated, c]):
            raise ValueError('Zero-weight channel changed')
    result[isolated] = tree[isolated]
    if not np.array_equal(result[isolated], tree[isolated]):
        raise ValueError('Isolated fallback changed')
    return result


def predict_bundle(bundle, recipe):
    return mix(bundle['T0'], bundle['T1'], bundle['LightGBM'],
               bundle['context'].isolated.to_numpy(bool), recipe['beta'],
               recipe.get('speed', 0.0), recipe.get('flow', 0.0))


def details(bundles, recipe, method, date_subset=None):
    rows = []
    for panel, bundle in bundles.items():
        ctx = bundle['context']
        keep = np.ones(len(ctx), bool) if date_subset is None else ctx.date.isin(date_subset).to_numpy()
        values = predict_bundle(bundle, recipe)[keep].copy()
        values[:, 1] /= ctx.lanes.to_numpy()[keep]
        rows.extend(metric_rows(bundle['truth'][keep], values, ctx.mask_regime.to_numpy()[keep],
                                method, panel, panel.rsplit('_', 1)[0], require_all=date_subset is None))
    return pd.DataFrame(rows)


def summarize(detail):
    result = aggregate(detail)
    return result.merge(detail.groupby('method', as_index=False)[['rmse_speed_kmh', 'rmse_flow_per_lane']].mean(), on='method')


def choose(table, priorities):
    """Deterministic 1e-12 tie handling, no hidden holdout-dependent choices."""
    if table.method.duplicated().any() or not np.isfinite(table.S_state).all():
        raise ValueError('Invalid selection table')
    best = table.S_state.max()
    tied = set(table.loc[table.S_state >= best-1e-12, 'method'])
    return min(tied, key=lambda name: priorities[name])


def select(bundles, dates):
    if any(set(b['context'].date) - set(dates) for b in bundles.values()):
        raise ValueError('Selection accessed dates outside internal validation')
    neural_details = pd.concat([details(bundles, {'beta': beta}, name) for name, beta in NEURAL.items()], ignore_index=True)
    neural_table = summarize(neural_details)
    chosen_neural = choose(neural_table, {name: i for i, name in enumerate(NEURAL)})
    beta = NEURAL[chosen_neural]
    recipes = {f'F_v{int(v*100):02d}_q{int(q*100):02d}': {'beta': beta, 'speed': v, 'flow': q}
               for v in WEIGHTS for q in WEIGHTS}
    grid_details = pd.concat([details(bundles, recipe, name) for name, recipe in recipes.items()], ignore_index=True)
    grid_table = summarize(grid_details)
    priorities = {name: (r['speed']+r['flow'], r['speed'], r['flow']) for name, r in recipes.items()}
    chosen = choose(grid_table, priorities)
    stability = []
    halves = {'early': dates[:len(dates)//2], 'late': dates[len(dates)//2:]}
    # Diagnostics only; selection above has already finished.
    for label, subset in halves.items():
        if not subset:
            continue
        for name, recipe in {**{n: {'beta': b} for n, b in NEURAL.items()}, 'Selected': recipes[chosen]}.items():
            frame = details(bundles, recipe, name, subset)
            if not frame.empty:
                summary = summarize(frame)
                summary['period'] = label
                summary['dates'] = ','.join(subset)
                stability.append(summary)
    selection = dict(neural=chosen_neural, grid_method=chosen, recipe=recipes[chosen],
                     selection_dates=dates, scope='Two directions, shared weights; internal labels only',
                     tie_rule='1e-12; neural T0/equal/T1; grid least total tree weight then speed weight')
    return selection, neural_details, grid_details, pd.concat(stability, ignore_index=True)


def evaluate(bundle, recipe, method):
    ctx = bundle['context'].copy()
    values = predict_bundle(bundle, recipe)
    normalized = values.copy()
    normalized[:, 1] /= ctx.lanes.to_numpy()
    truth = bundle['truth']
    ctx['speed_sq_error'] = (normalized[:, 0]-truth[:, 0])**2
    ctx['flow_lane_sq_error'] = (normalized[:, 1]-truth[:, 1])**2
    ctx['congested'] = truth[:, 0] <= .6*ctx.free_speed_kmh.to_numpy()
    groups = {}
    for name, columns in [('daily', ['date', 'mask_regime']), ('gap', ['gap']), ('flow_gap', ['flow_gap']),
                          ('regime_gap', ['mask_regime', 'gap']), ('congestion', ['congested']), ('isolation', ['isolated'])]:
        frame = ctx.groupby(columns, as_index=False).agg(n=('speed_sq_error', 'size'),
                    speed_mse=('speed_sq_error', 'mean'), flow_lane_mse=('flow_lane_sq_error', 'mean'))
        frame['method'] = method
        frame['panel'] = ctx.panel.iloc[0]
        frame['family'] = ctx.panel.iloc[0].rsplit('_', 1)[0]
        groups[name] = frame
    return values, groups
