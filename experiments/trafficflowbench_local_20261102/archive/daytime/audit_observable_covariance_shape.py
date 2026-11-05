"""Measure theoretical distinguishability of covariance laws using counts."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from itertools import permutations

import common as d
from audit_direct_od_measurement_design import base


def main():
    out = d.OUT / 'task4_observable_covariance_shape'
    d.save(out / 'protocol.json', dict(code_sha256=d.sha(__file__), split='private',
        weighting_split=['train', 'validation'], powers=[0., .5, 1.], shared_fractions=[.25, .5],
        method='Compare covariance shapes of idealized centered Gaussian count errors for allocations 0, .5, 1, with a fitted global scale as a nuisance parameter. No draws or observed error residuals.'))
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
        count_values, count_vectors = eigh(counts @ counts.T)
        keep = count_values > count_values[-1] * 1e-10
        count_basis = count_vectors[:, keep].T @ counts
        weights = .5 * (train / train.sum() + valid / valid.sum())
        n = len(paths)
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
                for true_share, comparison_share in permutations(covariances, 2):
                    spectrum = eigh(covariances[true_share], covariances[comparison_share], eigvals_only=True)
                    rank = len(spectrum)
                    fitted_scale = spectrum.mean()
                    normalized = spectrum/fitted_scale
                    raw_kl = .5 * np.sum(spectrum - 1 - np.log(spectrum))
                    adjusted_kl = .5 * np.sum(normalized - 1 - np.log(normalized))
                    assert -1e-10 <= adjusted_kl <= raw_kl + 1e-10
                    rows.append(dict(panel=panel, power=power, shared_fraction=shared_fraction,
                        true_origin_fraction=true_share, comparison_origin_fraction=comparison_share,
                        count_rank=rank, scale=fitted_scale, normalized_eigen_min=normalized.min(),
                        normalized_eigen_max=normalized.max(), raw_kl=raw_kl,
                        adjusted_kl=adjusted_kl, adjusted_kl_per_coordinate=adjusted_kl/rank))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_panel.csv', index=False)
    frame.groupby(['power', 'shared_fraction','true_origin_fraction','comparison_origin_fraction'])[['raw_kl','adjusted_kl','adjusted_kl_per_coordinate']].mean().to_csv(out / 'summary.csv')
    d.save(out / 'input_sha256.json', hashes)
    d.save(out / 'failures.json', [])


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
