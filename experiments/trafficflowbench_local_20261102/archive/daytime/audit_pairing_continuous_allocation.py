"""Find worst-case residual pairing variance across continuous allocation."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import brentq

import common as d
from audit_direct_od_measurement_design import base


def conditional_curve(constraints, diagonal, origin_factors, destination_factors, weights):
    cross_independent = diagonal[:, None] * constraints.T
    origin_cross = origin_factors @ (constraints @ origin_factors).T
    destination_cross = destination_factors @ (constraints @ destination_factors).T
    cross_center = cross_independent + .5 * (origin_cross + destination_cross)
    cross_direction = origin_cross - destination_cross
    values, vectors = eigh(constraints @ cross_center)
    keep = values > values[-1] * 1e-10
    whiten = vectors[:, keep] / np.sqrt(values[keep])
    slopes, rotate = eigh(whiten.T @ constraints @ cross_direction @ whiten)
    coordinates = whiten @ rotate
    left = cross_center @ coordinates
    right = cross_direction @ coordinates
    quadratic = weights @ (right**2)
    linear = 2 * (weights @ (left*right))
    constant = weights @ (left**2)
    total = weights @ (diagonal + np.sum(origin_factors**2, axis=1))

    def evaluate(share):
        x = share - .5
        denom = 1 + x * slopes
        numerator = constant + x * (linear + x * quadratic)
        trace = total - np.sum(numerator / denom)
        derivative = -np.sum(((linear + 2*x*quadratic)*denom - numerator*slopes)/denom**2)
        return trace, derivative

    return evaluate


def main():
    out = d.OUT / 'task4_pairing_continuous_allocation'
    d.save(out / 'protocol.json', dict(code_sha256=d.sha(__file__), split='private',
        weighting_split=['train', 'validation'], powers=[0., .5, 1.], shared_fractions=[.25, .5],
        method='Hold total and shared variance fixed in every cell. Use a symmetric constraint eigendecomposition to calculate the continuous conditional trace curve and maximize it using its derivative. No draws.'))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        paths, counts, _, _, h = base.load(panel, 'private')
        train_paths, _, _, train, ht = base.load(panel, 'train')
        valid_paths, _, _, valid, hv = base.load(panel, 'validation')
        assert paths.path_id.equals(train_paths.path_id) and paths.path_id.equals(valid_paths.path_id)
        for source in (h, ht, hv):
            hashes.update(source)
        origins, destinations = [pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                                 for key in ('origin_zone', 'destination_zone')]
        groups = np.vstack((origins, destinations))
        stacked = np.vstack((counts, groups))
        weights = .5 * (train / train.sum() + valid / valid.sum())
        n = len(paths)
        for power in (0., .5, 1.):
            variance = np.maximum(weights, .1 / n)**power
            variance /= variance.mean()
            for shared_fraction in (.25, .5):
                diagonal = (1 - shared_fraction) * variance
                sd = np.sqrt(shared_fraction * variance)
                curve = conditional_curve(stacked, diagonal, sd[:, None]*origins.T, sd[:, None]*destinations.T, weights)
                grid = np.linspace(0., 1., 5)
                values = np.array([curve(share)[0] for share in grid])
                first, last = curve(0.)[1], curve(1.)[1]
                assert first >= last - 1e-9
                if max(abs(first), abs(last)) < 1e-9:
                    best = .5
                elif first <= 0:
                    best = 0.
                elif last >= 0:
                    best = 1.
                else:
                    best = brentq(lambda share: curve(share)[1], 0., 1.)
                peak = curve(best)[0]
                assert peak >= values.max() - 1e-9
                rows.append(dict(panel=panel, power=power, shared_fraction=shared_fraction,
                    maximum_origin_fraction=best, maximum_trace=peak, grid_maximum=values.max(),
                    grid_minimum=values.min(), continuous_minimum=min(values[0], values[-1]),
                    **{f'grid_{i}': value for i, value in enumerate(values)}))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_panel.csv', index=False)
    frame.groupby(['power', 'shared_fraction'])[['maximum_origin_fraction', 'maximum_trace', 'grid_maximum', 'continuous_minimum']].mean().to_csv(out / 'summary.csv')
    d.save(out / 'input_sha256.json', hashes)
    d.save(out / 'failures.json', [])


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
