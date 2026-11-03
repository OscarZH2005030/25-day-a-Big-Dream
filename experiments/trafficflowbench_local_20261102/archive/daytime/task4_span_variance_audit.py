"""Fixed-family tests of OD-span-dependent prior error variance, not OD accuracy."""
import common as d
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd
import json
import msvcrt
import traceback
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from datetime import datetime, timezone

O = d.OUT / 'task4_span_variance_audit'
MODELS = {'uniform': 0., 'short_variance': -2., 'long_variance': 2., 'learned_span': None}


def record(panel, split):
    paths, A, c, b, hashes = base.load(panel, split)
    cache = d.WORK / 'task4_covariance_inputs' / panel / (split + '.npz')
    z = np.load(cache)
    A, c = A[z['selected_count_rows']], c[z['selected_count_rows']]
    loaded = A @ b
    scale = float(np.median(loaded[loaded > 0]))
    r = (c - loaded) / scale
    assert np.allclose(r, z['r'], atol=1e-10, rtol=0)
    origin = paths.origin_zone.str.extract(r'^Z(\d+)$', expand=False).astype(int).to_numpy()
    destination = paths.destination_zone.str.extract(r'^Z(\d+)$', expand=False).astype(int).to_numpy()
    span = destination - origin
    t = (span - span.mean()) / span.max()
    return dict(panel=panel, family=panel.rsplit('_', 1)[0], A=A, r=r, t=t, rank=len(r), hashes=hashes, cache_sha256=d.sha(cache))


def quadratic(rec, eta):
    w = np.exp(eta * rec['t'])
    C = (rec['A'] * w) @ rec['A'].T
    C /= np.trace(C) / rec['rank']
    C += np.eye(rec['rank']) * 1e-6
    factor = cho_factor(C, lower=True, check_finite=False)
    logdet = 2 * np.log(np.diag(factor[0])).sum() / rec['rank']
    q = rec['r'] @ cho_solve(factor, rec['r'], check_finite=False) / rec['rank']
    return float(q), float(logdet)


def profile(records, eta):
    values = np.array([quadratic(rec, eta) for rec in records])
    variance = max(float(values[:, 0].mean()), 1e-10)
    loss = .5 * (np.log(2 * np.pi * variance) + values[:, 1].mean() + values[:, 0].mean() / variance)
    return float(loss), variance


def fit(records, name):
    eta = MODELS[name]
    diagnostics = None
    if eta is None:
        grid = np.linspace(-4, 4, 17)
        losses = [profile(records, value)[0] for value in grid]
        index = int(np.argmin(losses))
        left, right = grid[max(index - 1, 0)], grid[min(index + 1, 16)]
        search = minimize_scalar(lambda value: profile(records, value)[0], bounds=(left, right), method='bounded', options={'xatol': 1e-7})
        candidates = [(float(grid[index]), losses[index]), (float(search.x), float(search.fun))]
        eta, _ = min(candidates, key=lambda item: item[1])
        diagnostics = dict(grid_eta=grid.tolist(), grid_losses=losses, optimizer_success=bool(search.success), optimizer_message=str(search.message))
    loss, variance = profile(records, eta)
    return dict(model=name, eta=float(eta), variance=variance, train_NLL_per_dimension=loss, diagnostics=diagnostics)


def evaluate(rec, model):
    q, logdet = quadratic(rec, model['eta'])
    standardized = q / model['variance']
    return dict(NLL_per_dimension=float(.5 * (np.log(2 * np.pi * model['variance']) + logdet + standardized)), whitened_quadratic_per_dimension=standardized)


def main():
    O.mkdir(parents=True, exist_ok=True)
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(), hypothesis='Uniform independent additive path errors may miss span-dependent uncertainty; compare fixed and train-fitted variance shapes across corridor families before any OD inference.', models=MODELS, mean='Zero mean fixed: preceding mean audit favored zero mean in cross-family CV; no background subtraction or GLS scaling fitted here.', covariance='A diag(exp(eta*(zone_span-mean_span)/max_span)) A.T; each road normalized to mean diagonal1, jitter1e-6, one shared profiled variance.', learned_search='17-point grid on[-4,4], bounded refinement only between neighboring grid points, retain better grid/refinement candidate; training objective only.', validation='Five leave-one-corridor-family folds on public train; freeze full-train models before reporting all supplied validation/private public residuals.', seed='No randomness; fixed four BLAS threads.', source_sha256=d.sha(__file__), scope='Exposed public count-space development diagnostic, not hidden OD validation. Covariance does not identify arbitrary nullspace error; no path candidate produced and official baseline unchanged.')
    d.save(O / 'protocol.json', protocol)
    records = [record(panel, 'train') for panel in d.night.predictor.load_routes()]
    d.save(O / 'train_input_sha256.json', {key: value for rec in records for key, value in rec['hashes'].items()})
    rows, models = [], {}
    for family in sorted({r['family'] for r in records}):
        train, test = [r for r in records if r['family'] != family], [r for r in records if r['family'] == family]
        for name in MODELS:
            d.state(status='running', stage='task4_span_variance_family_cv', family=family, model=name, next='Count-space uncertainty mechanism only; keep all negative results, no OD score claim.')
            model = fit(train, name)
            models[family + '/' + name] = model
            rows.extend(dict(panel=r['panel'], family=family, model=name, **evaluate(r, model)) for r in test)
        pd.DataFrame(rows).to_csv(O / 'family_cv_by_panel.csv', index=False)
        d.save(O / 'family_cv_models.json', models)
    pd.DataFrame(rows).groupby('model')[['NLL_per_dimension', 'whitened_quadratic_per_dimension']].mean().to_csv(O / 'family_cv_summary.csv')
    frozen = {name: fit(records, name) for name in MODELS}
    d.save(O / 'full_train_models.json', dict(models=frozen, frozen_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=d.sha(O / 'protocol.json')))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        for split in ['validation', 'private']:
            rec = record(panel, split)
            hashes.update(rec['hashes'])
            rows.extend(dict(panel=panel, split=split, model=name, **evaluate(rec, model)) for name, model in frozen.items())
    pd.DataFrame(rows).to_csv(O / 'transfer_by_panel.csv', index=False)
    pd.DataFrame(rows).groupby(['split', 'model'])[['NLL_per_dimension', 'whitened_quadratic_per_dimension']].mean().to_csv(O / 'transfer_summary.csv')
    d.save(O / 'transfer_input_sha256.json', hashes)
    d.state(status='stage_complete', stage='task4_span_variance_audit_complete', next='Inspect variance-shape transport and cross-check uniform mean-zero control; no hidden OD inference or Kaggle upload.')


if __name__ == '__main__':
    lock = (d.ROOT / 'work/overnight/research.lock').open('r+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        main()
    except Exception as error:
        d.state(status='failed', stage='task4_span_variance_audit', error=repr(error), next='Retain failure and fixed protocol; no change to official baseline.')
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
