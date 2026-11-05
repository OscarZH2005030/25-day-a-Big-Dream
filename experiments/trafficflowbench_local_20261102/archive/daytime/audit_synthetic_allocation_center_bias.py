"""Recover shared error allocation from synthetic centered count draws."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.linalg import cholesky, solve_triangular

import common as d
from audit_direct_od_measurement_design import base


def independent_count_rows(counts):
    # Fix the original observation coordinates for seeded draws.
    orthogonal = []
    selected = []
    for i, row in enumerate(counts):
        residual = row.astype(float)
        for _ in range(2):
            for vector in orthogonal:
                residual -= (residual @ vector) * vector
        norm = np.linalg.norm(residual)
        if norm > 1e-8:
            orthogonal.append(residual / norm)
            selected.append(i)
    return counts[selected]


def main():
    out = d.OUT / 'task4_synthetic_allocation_center_bias'
    d.save(out / 'protocol.json', dict(code_sha256=d.sha(__file__), split='private',
        weighting_split=['train', 'validation'], powers=[0., .5, 1.], shared_fractions=[.25, .5],
        seed=2026110501, draws=512, count_basis="First linearly independent original rows in input order",
        method='Reuse centered Gaussian draws, add mean shifts along the pooled-prior count profile at 0, .5 and 1 whitened RMS, and compare profile likelihood before and after removing the known shift direction.'))
    rows, hashes = [], {}
    rng = np.random.default_rng(2026110501)
    for panel in sorted(d.night.predictor.load_routes()):
        paths, counts, _, _, h = base.load(panel, 'private')
        train_paths, _, _, train, ht = base.load(panel, 'train')
        valid_paths, _, _, valid, hv = base.load(panel, 'validation')
        assert paths.path_id.equals(train_paths.path_id) and paths.path_id.equals(valid_paths.path_id)
        for source in (h, ht, hv):
            hashes.update(source)
        origins, destinations = [pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                                 for key in ('origin_zone', 'destination_zone')]
        count_basis = independent_count_rows(counts)
        weights = .5 * (train / train.sum() + valid / valid.sum())
        n = len(paths)
        models = {}
        for power in (0., .5, 1.):
            variance = np.maximum(weights, .1 / n)**power
            variance /= variance.mean()
            for shared_fraction in (.25, .5):
                diagonal = (1 - shared_fraction) * variance
                sd = np.sqrt(shared_fraction * variance)
                base_covariance = (count_basis * diagonal) @ count_basis.T
                projected_origin = (count_basis * sd) @ origins.T
                projected_destination = (count_basis * sd) @ destinations.T
                origin_covariance = projected_origin @ projected_origin.T
                destination_covariance = projected_destination @ projected_destination.T
                covariances = {share: base_covariance + share*origin_covariance + (1-share)*destination_covariance
                               for share in (0., .5, 1.)}
                for share, covariance in covariances.items():
                    factor = cholesky(covariance, lower=True)
                    logdet = 2*np.log(factor.diagonal()).sum()
                    models[power, shared_fraction, share] = factor, logdet
        candidates = list(models)
        rank = count_basis.shape[0]
        direction = count_basis @ weights
        contrasts = null_space(direction[None, :]).T
        np.testing.assert_allclose(contrasts @ direction, 0, atol=1e-12, rtol=0)
        contrast_models = []
        for factor, _ in models.values():
            projected = contrasts @ factor
            reduced = cholesky(projected @ projected.T, lower=True)
            contrast_models.append((reduced, 2*np.log(reduced.diagonal()).sum()))
        for true_key, (factor, _) in models.items():
            power, shared_fraction, share = true_key
            samples = factor @ rng.standard_normal((rank, 512))
            shift = direction * np.sqrt(rank) / np.linalg.norm(solve_triangular(factor, direction, lower=True))
            for amplitude in (0., .5, 1.):
                observed = samples + amplitude * shift[:, None]
                for regime, observed_space, model_space in (
                    ('unadjusted', observed, list(models.values())),
                    ('contrast', contrasts @ observed, contrast_models),
                ):
                    dimension = observed_space.shape[0]
                    scores = []
                    for candidate_factor, logdet in model_space:
                        whitened = solve_triangular(candidate_factor, observed_space, lower=True)
                        scores.append(logdet + dimension*np.log(np.mean(whitened**2, axis=0)))
                    predicted = np.array([candidates[i][2] for i in np.argmin(scores,axis=0)])
                    for predicted_share in (0., .5, 1.):
                        rows.append(dict(panel=panel, power=power, shared_fraction=shared_fraction,
                            true_origin_fraction=share, predicted_origin_fraction=predicted_share,
                            regime=regime, amplitude=amplitude, count_rank=rank, draws=512,
                            matches=int(np.sum(predicted==predicted_share))))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_panel.csv', index=False)
    keys = ['regime', 'amplitude', 'power', 'shared_fraction', 'true_origin_fraction', 'predicted_origin_fraction']
    frame.groupby(keys)[['matches', 'draws']].sum().to_csv(out / 'summary.csv')
    d.save(out / 'input_sha256.json', hashes)
    d.save(out / 'failures.json', [])


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
