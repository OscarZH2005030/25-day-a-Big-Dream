"""Compare group error shapes at equal prior-weighted shared variance."""
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
    out = d.OUT / 'task4_pairing_matched_group_strength'
    d.save(out / 'protocol.json', dict(code_sha256=d.sha(__file__), split='private',
        weighting_split=['train', 'validation'], powers=[0., .5, 1.], correlations=[.125, .25],
        method='Hold the covariance diagonal fixed. Compare cell-scaled group errors to group-specific constants with variance rho times the smallest cell variance of the group. Match prior-weighted shared variance across factors with one global rescaling per road and setting, then adjust independent variance to keep each cell total variance fixed. Condition via symmetric constraint Gram matrices. No draws.'))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        paths, counts, _, _, h = base.load(panel, 'private')
        train_paths, _, _, train, ht = base.load(panel, 'train')
        valid_paths, _, _, valid, hv = base.load(panel, 'validation')
        assert paths.path_id.equals(train_paths.path_id) and paths.path_id.equals(valid_paths.path_id)
        for source in (h, ht, hv):
            hashes.update(source)
        groups = np.vstack([pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                            for key in ('origin_zone', 'destination_zone')])
        stacked = np.vstack((counts, groups))
        weights = .5 * (train / train.sum() + valid / valid.sum())
        n = len(paths)
        for power in (0., .5, 1.):
            variance = np.maximum(weights, .1 / n)**power
            variance /= variance.mean()
            group_minimum = np.array([variance[row.astype(bool)].min() for row in groups])
            for rho in (.125, .25):
                aligned = groups.T * np.sqrt(rho * group_minimum)
                scaled = np.sqrt(rho * variance)[:, None] * groups.T
                shared_budget = weights @ np.sum(aligned**2, axis=1)
                scale = np.sqrt(shared_budget / (weights @ np.sum(scaled**2, axis=1)))
                assert 0 < scale <= 1 + 1e-12
                for law, factors in (('cell_scaled', scaled * scale), ('group_aligned', aligned)):
                    shared = np.sum(factors**2, axis=1)
                    np.testing.assert_allclose(weights @ shared, shared_budget, rtol=0, atol=1e-12)
                    diagonal = variance - shared
                    assert diagonal.min() > 0
                    count_var, count_rank = conditioned_diagonal(counts, diagonal, factors)
                    pairing_var, total_rank = conditioned_diagonal(stacked, diagonal, factors)
                    assert pairing_var.min() >= -1e-8
                    idiosyncratic_var, _ = conditioned_diagonal(stacked, diagonal)
                    shared_increment = weights @ (pairing_var - idiosyncratic_var)
                    assert shared_increment >= -1e-9
                    rows.append(dict(panel=panel, power=power, rho=rho, law=law,
                        count_rank=count_rank, residual_nullity=n-total_rank, scale=scale,
                        shared_trace=weights @ shared, shared_increment=shared_increment,
                        count_trace=weights @ count_var,
                        pairing_trace=weights @ pairing_var,
                        remaining_share=(weights @ pairing_var)/(weights @ count_var)))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_panel.csv', index=False)
    frame.groupby(['power', 'rho', 'law'])[['shared_trace', 'shared_increment', 'count_trace', 'pairing_trace', 'remaining_share']].mean().to_csv(out / 'summary.csv')
    d.save(out / 'input_sha256.json', hashes)
    d.save(out / 'failures.json', [])


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
