"""Follow-up to a boundary span-variance fit: range and correlated-demand controls."""
import common as d
import task4_span_variance_audit as parent
import numpy as np
import pandas as pd
import msvcrt
import traceback
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar
from datetime import datetime, timezone

O = d.OUT / 'task4_variance_controls_audit'
MODELS = ['uniform', 'span4', 'span8', 'span12', 'extended_span', 'global_demand_mixture']


def record(panel, split):
    rec = parent.record(panel, split)
    z = np.load(d.WORK / 'task4_covariance_inputs' / panel / (split + '.npz'))
    rec.update(l=z['l'], iid=z['Cs'][1])
    return rec


def quadratic(rec, kind, parameter):
    if kind == 'span':
        w = np.exp(parameter * rec['t'])
        C = (rec['A'] * w) @ rec['A'].T
    else:
        factor = np.outer(rec['l'], rec['l'])
        factor /= np.trace(factor) / rec['rank']
        C = (1 - parameter) * rec['iid'] + parameter * factor
    C /= np.trace(C) / rec['rank']
    C += np.eye(rec['rank']) * 1e-6
    factor = cho_factor(C, lower=True, check_finite=False)
    return float(rec['r'] @ cho_solve(factor, rec['r'], check_finite=False) / rec['rank']), float(2 * np.log(np.diag(factor[0])).sum() / rec['rank'])


def profile(records, kind, parameter):
    values = np.array([quadratic(r, kind, parameter) for r in records])
    variance = max(float(values[:, 0].mean()), 1e-10)
    return float(.5 * (np.log(2 * np.pi * variance) + values[:, 1].mean() + 1)), variance


def fit(records, name):
    diagnostics = None
    kind = 'global' if name == 'global_demand_mixture' else 'span'
    if name in ['extended_span', 'global_demand_mixture']:
        grid = np.linspace(-4, 12, 17) if kind == 'span' else np.array([0, .1, .25, .5, .75, .9, .99, .9999])
        losses = [profile(records, kind, value)[0] for value in grid]
        i = int(np.argmin(losses))
        opt = minimize_scalar(lambda value: profile(records, kind, value)[0], bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method='bounded', options={'xatol': 1e-7})
        parameter, _ = min([(float(grid[i]), losses[i]), (float(opt.x), float(opt.fun))], key=lambda item: item[1])
        diagnostics = dict(grid=grid.tolist(), losses=losses, optimizer_success=bool(opt.success), boundary=bool(parameter in [float(grid[0]), float(grid[-1])]))
    else:
        parameter = {'uniform': 0., 'span4': 4., 'span8': 8., 'span12': 12.}[name]
    loss, variance = profile(records, kind, parameter)
    return dict(model=name, kind=kind, parameter=parameter, variance=variance, train_NLL_per_dimension=loss, diagnostics=diagnostics)


def evaluate(rec, model):
    q, logdet = quadratic(rec, model['kind'], model['parameter'])
    return dict(NLL_per_dimension=float(.5 * (np.log(2 * np.pi * model['variance']) + logdet + q / model['variance'])), whitened_quadratic_per_dimension=q / model['variance'])


def main():
    O.mkdir(parents=True, exist_ok=True)
    d.save(O / 'protocol.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(), trigger='Previous span-variance audit hit eta4 upper bound in all five train folds and full fit; its validation/private residuals are already exposed development results.', hypothesis='Long-span heteroscedastic covariance may improve count residual likelihood, but eta4 may be only a boundary artifact or proxy for shared demand-level error.', models=MODELS, extended_search='17-point eta grid[-4,12], bounded local refinement around best point; keep better grid/refinement. Fixed4/8/12 controls retained.', demand_control='Normalized convex covariance mixture of additive independent path covariance and rank-one (Ab)(Ab).T, weight0..0.9999 fit on training roads; random global demand-scale error control, not direct OD truth.', mean='Zero fixed for every model; one shared variance profiled only on training roads.', validation='Five public-train leave-family folds; full models frozen before evaluating supplied validation/private public inputs. This is follow-up development, not fresh blind data. All six models reported, no path candidate or official gain claim.', seed='Deterministic, four BLAS threads.', source_sha256=d.sha(__file__), parent_source_sha256=d.sha(d.WORK / 'task4_span_variance_audit.py'), prior_protocol_sha256=d.sha(d.OUT / 'task4_span_variance_audit/protocol.json'), scope='Observable count uncertainty alone does not identify OD pairing or nullspace errors. Official Task4 baseline unchanged, Kaggle uploads prohibited.'))
    records = [record(panel, 'train') for panel in d.night.predictor.load_routes()]
    d.save(O / 'train_input_sha256.json', {key: value for rec in records for key, value in rec['hashes'].items()})
    rows, models = [], {}
    for family in sorted({r['family'] for r in records}):
        train = [r for r in records if r['family'] != family]
        for name in MODELS:
            d.state(status='running', stage='task4_variance_controls_cv', family=family, model=name, next='Check range-boundary and global-demand alternatives before interpreting long-span count covariance; no official score claim.')
            model = fit(train, name)
            models[family + '/' + name] = model
            rows.extend(dict(panel=r['panel'], family=family, model=name, **evaluate(r, model)) for r in records if r['family'] == family)
        pd.DataFrame(rows).to_csv(O / 'family_cv_by_panel.csv', index=False)
        d.save(O / 'family_cv_models.json', models)
    pd.DataFrame(rows).groupby('model')[['NLL_per_dimension', 'whitened_quadratic_per_dimension']].mean().to_csv(O / 'family_cv_summary.csv')
    models = {name: fit(records, name) for name in MODELS}
    d.save(O / 'full_train_models.json', dict(models=models, frozen_utc=datetime.now(timezone.utc).isoformat()))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        for split in ['validation', 'private']:
            rec = record(panel, split)
            hashes.update(rec['hashes'])
            rows.extend(dict(panel=panel, split=split, model=name, **evaluate(rec, model)) for name, model in models.items())
    pd.DataFrame(rows).to_csv(O / 'transfer_by_panel.csv', index=False)
    pd.DataFrame(rows).groupby(['split', 'model'])[['NLL_per_dimension', 'whitened_quadratic_per_dimension']].mean().to_csv(O / 'transfer_summary.csv')
    d.save(O / 'transfer_input_sha256.json', hashes)
    d.state(status='stage_complete', stage='task4_variance_controls_complete', next='Verify all controls, reproduce standalone and publish only substantiated count-space results with OD nonidentifiability caveats; do not replace official baseline.')


if __name__ == '__main__':
    lock = (d.ROOT / 'work/overnight/research.lock').open('r+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        main()
    except Exception as error:
        d.state(status='failed', stage='task4_variance_controls', error=repr(error), next='Preserve negative results and registered range/model controls.')
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
