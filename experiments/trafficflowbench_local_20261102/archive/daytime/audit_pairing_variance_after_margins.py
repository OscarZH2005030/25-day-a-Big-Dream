"""Measure residual OD pairing uncertainty even with exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base


def main():
    out = d.OUT / 'task4_pairing_variance_after_margins'
    powers = (0., .5, 1.)
    correlations = (0., .125, .25)
    d.save(out / 'protocol.json', dict(code_sha256=d.sha(__file__), split='private',
        weighting_split=['train', 'validation'], powers=powers, correlations=correlations,
        method='Condition matched diagonal variances and shared origin/destination errors on exact counts, then exact complete margins. Compare sequential and stacked projections in the factor space. No draws.'))
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
        weights = .5 * (train / train.sum() + valid / valid.sum())
        n = len(paths)
        for power in powers:
            variance = np.maximum(weights, .1 / n)**power
            variance /= variance.mean()
            for rho in correlations:
                sd = np.sqrt(variance)
                diagonal_factor = np.diag(sd * np.sqrt(1 - 2 * rho))
                factor = np.column_stack((diagonal_factor, sd[:, None] * groups.T * np.sqrt(rho))) if rho else diagonal_factor
                np.testing.assert_allclose(np.sum(factor**2, axis=1), variance, atol=1e-12, rtol=0)
                count_basis = orth((counts @ factor).T, rcond=1e-10)
                after_counts = factor - (factor @ count_basis) @ count_basis.T
                margin_basis = orth((groups @ after_counts).T, rcond=1e-10)
                pairing = after_counts - (after_counts @ margin_basis) @ margin_basis.T
                stacked = np.vstack((counts, groups))
                direct_basis = orth((stacked @ factor).T, rcond=1e-10)
                direct = factor - (factor @ direct_basis) @ direct_basis.T
                np.testing.assert_allclose(pairing, direct, atol=1e-9, rtol=0)
                before_trace, count_trace, pairing_trace = [
                    weights @ np.sum(matrix**2, axis=1) for matrix in (factor, after_counts, pairing)]
                assert 0 < pairing_trace <= count_trace <= before_trace + 1e-10
                rows.append(dict(panel=panel, power=power, rho=rho,
                    count_rank=count_basis.shape[1], margin_rank=margin_basis.shape[1],
                    residual_nullity=n-direct_basis.shape[1], prior_trace=before_trace,
                    count_trace=count_trace, pairing_trace=pairing_trace,
                    remaining_share=pairing_trace/count_trace))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_panel.csv', index=False)
    frame.groupby(['power', 'rho'])[['count_trace', 'pairing_trace', 'remaining_share']].mean().to_csv(out / 'summary.csv')
    d.save(out / 'input_sha256.json', hashes)
    d.save(out / 'failures.json', [])


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
