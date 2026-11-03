"""Fresh fixed-seed artificial stress of span-dependent uncertainty penalties."""
import common as d
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd
import json
import hashlib
import msvcrt
import traceback
from datetime import datetime, timezone
from scipy.linalg import solve

O = d.OUT / 'task4_weighted_variance_stress'
SEED = 2026110312
TRUTHS = base.TRUTHS
NOISES = ['additive_clipped', 'multiplicative', 'origin_destination', 'positive_background', 'span_heterogeneous_clipped']
METHODS = {'baseline': 0., 'variance010': .1, 'variance025': .25, 'variance100': 1.}


def ridge(A, c, b, weight):
    lam, y = .05, np.zeros(len(c))
    def fg(y):
        x = np.maximum(b - weight * (A.T @ y) / lam, 0)
        return .5 * lam * np.sum(x * x / weight) + c @ y + .5 * (y @ y), y + c - A @ x, x
    for iteration in range(150):
        value, gradient, x = fg(y)
        if np.abs(gradient).max() < 1e-7:
            break
        H = (A * (weight * (x > 0)) / lam) @ A.T + np.eye(len(c))
        step = solve(H, -gradient, assume_a='pos')
        rate, slope = 1., gradient @ step
        for _ in range(60):
            if fg(y + rate * step)[0] <= value + 1e-4 * rate * slope + 1e-8:
                break
            rate *= .5
        y += rate * step
    _, gradient, x = fg(y)
    return x, bool(np.abs(gradient).max() < 1e-5)


def main():
    O.mkdir(parents=True, exist_ok=True)
    model_path = d.OUT / 'task4_variance_controls_audit/full_train_models.json'
    models = json.loads(model_path.read_text())['models']
    eta = models['extended_span']['parameter']
    variance = models['uniform']['variance']
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, hypothesis='A span-dependent variance shape supported by public train count covariance may improve weighted OD inference, but observable likelihood cannot rule out harmful nullspace extrapolation. Stress all fixed strengths under matched-shape and wrong-noise cases.', truths=TRUTHS, noises=NOISES, levels=['low','high'], replicates=2, panels=10, case_count=1200, eta_frozen_from_train=eta, uniform_noise_variance_frozen_from_train=variance, methods=METHODS, algorithm='Nonnegative weighted ridge, objective0.5||Af-c||2+0.5*0.05*sum((f-b)^2/w). w=(1-alpha)+alpha*w_span, w_span trace-normalized relative to uniform under original independent QR equations.', noise='Additive scales0.5/2 from uniform covariance fit; logsigma0.35/0.8, origin/destination0.25/0.6, positive lognormal background0.5/2 times truth mean. Span noise uses calibrated w_span then clips supplied b positive; clipping is explicit misspecification, not an exact posterior-model control.', selection='All four strengths retained, no selection on these artificial scores. Same public networks but new synthetic seed, not whole-system blind data or hidden OD validation.', source_sha256=d.sha(__file__), source_model_sha256=d.sha(model_path), scope='No actual Task4 candidate exported, official frozen reference unchanged and no Kaggle upload.')
    d.save(O / 'protocol.json', protocol)
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        paths, A, real_c, real_b, input_hashes = base.load(panel, 'train')
        hashes.update(input_hashes)
        cache = np.load(d.WORK / 'task4_covariance_inputs' / panel / 'train.npz')
        independent = A[cache['selected_count_rows']]
        oi = paths.origin_zone.str.extract(r'^Z(\d+)$', expand=False).astype(int).to_numpy()
        di = paths.destination_zone.str.extract(r'^Z(\d+)$', expand=False).astype(int).to_numpy()
        span = di - oi
        weight = np.exp(eta * (span - span.mean()) / span.max())
        weight *= np.sum(independent * independent) / np.sum(independent * independent * weight)
        loaded = independent @ real_b
        scale = float(np.median(loaded[loaded > 0]))
        sigma = np.sqrt(variance) * scale / np.sqrt(np.mean(np.sum(independent * independent, axis=1)))
        # The new solver must reduce to the old baseline at uniform weights.
        old, status = base.ridge(A, real_c, real_b)
        new, success = ridge(A, real_c, real_b, np.ones(len(real_b)))
        assert status['success'] and success and np.max(np.abs(old-new)) < 1e-6
        for truth_name in TRUTHS:
            for noise in NOISES:
                for level in ['low', 'high']:
                    for rep in range(2):
                        case = f'{panel}/{truth_name}/{noise}/{level}/{rep}'
                        seed = int.from_bytes(hashlib.sha256((str(SEED)+'/'+case).encode()).digest()[:8], 'little') % (2**63)
                        rng = np.random.default_rng(seed)
                        truth, origin, dest = base.truth_draw(paths, A, float(np.median(real_c[real_c > 0])), truth_name, rng)
                        factor = .5 if level == 'low' else 2.
                        if noise in ['additive_clipped', 'span_heterogeneous_clipped']:
                            sd = sigma * factor * (np.sqrt(weight) if noise.startswith('span_') else 1.)
                            b = np.maximum(truth + rng.normal(size=len(truth)) * sd, 1e-6)
                        elif noise == 'multiplicative':
                            sd = .35 if level == 'low' else .8
                            b = truth * np.exp(rng.normal(-sd*sd/2, sd, len(truth)))
                        elif noise == 'origin_destination':
                            sd = .25 if level == 'low' else .6
                            b = truth * np.exp(rng.normal(0, sd, origin.max()+1)[origin] + rng.normal(0, sd, dest.max()+1)[dest] - sd*sd)
                        else:
                            b = truth + factor * truth.mean() * rng.lognormal(-.7**2/2, .7, len(truth))
                        c = A @ truth
                        for name, alpha in METHODS.items():
                            prediction, success = ridge(A, c, b, (1-alpha)+alpha*weight)
                            row = dict(case=case, panel=panel, truth=truth_name, noise=noise, level=level, replicate=rep, seed=seed, method=name, solver_success=success)
                            if success:
                                row.update(base.metrics(prediction, truth, b, A, c, dest))
                            rows.append(row)
        pd.DataFrame(rows).to_csv(O / 'all_cases.csv', index=False)
        d.state(status='running', stage='task4_weighted_variance_stress', panel=panel, completed_rows=len(rows), next='Retain all fixed weighted-penalty harms, including OD component; no real candidate or Kaggle upload.')
    df = pd.DataFrame(rows)
    d.save(O / 'input_sha256.json', hashes)
    df.groupby(['truth','noise','level','method'])[[c for c in df if c.startswith('synthetic_')]].mean().to_csv(O / 'by_assumption.csv')
    baseline = df[df.method == 'baseline'].set_index('case')
    risk = []
    for name in METHODS:
        group = df[df.method == name].set_index('case')
        good = group.solver_success & baseline.solver_success
        delta = group.loc[good, 'synthetic_S_ODME'] - baseline.loc[good, 'synthetic_S_ODME']
        od = group.loc[good, 'synthetic_S_od'] - baseline.loc[good, 'synthetic_S_od']
        risk.append(dict(method=name, paired_cases=len(delta), mean_score_delta=float(delta.mean()), worst_score_delta=float(delta.min()), score_harmed_cases=int((delta < -1e-12).sum()), mean_OD_delta=float(od.mean()), OD_harmed_cases=int((od < -1e-12).sum()), solver_failed_cases=int((~group.solver_success).sum())))
    pd.DataFrame(risk).to_csv(O / 'paired_risk.csv', index=False)
    assert df.case.nunique() == protocol['case_count']
    d.state(status='stage_complete', stage='task4_weighted_variance_stress_complete', next='Audit solver failures and per-assumption harms, independently reproduce variance controls before publishing any support. Artificial gains cannot prove hidden OD gains; official reference unchanged.')


if __name__ == '__main__':
    lock = (d.ROOT / 'work/overnight/research.lock').open('r+b')
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        main()
    except Exception as error:
        d.state(status='failed', stage='task4_weighted_variance_stress', error=repr(error), next='Preserve fixed cases and failure; repair numerical issue without changing evaluation recipes.')
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
