"""Allocate a fixed shared covariance budget between origins and destinations."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

import common as d
from audit_direct_od_measurement_design import base


def conditioned_diagonal(constraints, diagonal, factors=None):
    cross = diagonal[:, None] * constraints.T
    if factors is not None:
        cross += factors @ (constraints @ factors).T
    inverse, rank = pinvh(constraints @ cross, rtol=1e-10, return_rank=True)
    prior = diagonal if factors is None else diagonal + np.sum(factors**2, axis=1)
    return prior - np.sum((cross @ inverse) * cross, axis=1), rank


def main():
    out = d.OUT / 'task4_pairing_origin_destination_allocation'
    d.save(out / 'protocol.json', dict(code_sha256=d.sha(__file__), split='private',
        weighting_split=['train', 'validation'], powers=[0., .5, 1.], shared_fractions=[.25, .5],
        method='Keep total and shared variance fixed in every cell. Allocate shared variance between origin and destination group factors at origin fractions 0, .25, .5, .75, 1. Condition via symmetric constraint Gram matrices. No draws.'))
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
                idiosyncratic_var, _ = conditioned_diagonal(stacked, diagonal)
                for origin_fraction in (0., .25, .5, .75, 1.):
                    scales = np.repeat([np.sqrt(origin_fraction), np.sqrt(1-origin_fraction)],
                                       [len(origins), len(destinations)])
                    factors = np.sqrt(shared_fraction * variance)[:, None] * groups.T * scales
                    shared = np.sum(factors**2, axis=1)
                    np.testing.assert_allclose(shared, shared_fraction * variance, rtol=0, atol=1e-12)
                    count_var, count_rank = conditioned_diagonal(counts, diagonal, factors)
                    pairing_var, total_rank = conditioned_diagonal(stacked, diagonal, factors)
                    assert pairing_var.min() >= -1e-8
                    shared_increment = weights @ (pairing_var - idiosyncratic_var)
                    assert shared_increment >= -1e-9
                    rows.append(dict(panel=panel, power=power, shared_fraction=shared_fraction, origin_fraction=origin_fraction,
                        count_rank=count_rank, residual_nullity=n-total_rank,
                        shared_trace=weights @ shared, shared_increment=shared_increment,
                        count_trace=weights @ count_var,
                        pairing_trace=weights @ pairing_var,
                        remaining_share=(weights @ pairing_var)/(weights @ count_var)))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_panel.csv', index=False)
    frame.groupby(['power', 'shared_fraction', 'origin_fraction'])[['shared_trace', 'shared_increment', 'count_trace', 'pairing_trace', 'remaining_share']].mean().to_csv(out / 'summary.csv')
    d.save(out / 'input_sha256.json', hashes)
    d.save(out / 'failures.json', [])


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
