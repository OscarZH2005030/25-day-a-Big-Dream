"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_component_pool_transfer'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__), predictor_split='train',
        coefficient_split='validation', evaluation_split='private',
        scope='Test whether train-to-validation component coefficients transfer to a two-source mean predictor.'))
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        train_paths, _, _, train, provenance=base.load(panel,'train')
        hashes.update(provenance)
        train=train/train.sum()
        for split in ('validation','private'):
            paths, counts, _, prior, provenance=base.load(panel,split)
            hashes.update(provenance)
            assert paths.path_id.equals(train_paths.path_id)
            target=prior/prior.sum()
            groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T
                              for k in ('origin_zone','destination_zone')])
            origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                                for k in ('origin_zone','destination_zone')]
            span=destination-origin
            bins=(span[:,None]==np.unique(span)).astype(float).T
            basis = orth(np.vstack((counts, groups)).T, rcond=1e-10)
            hist_basis = orth(np.vstack((counts, groups, bins)).T, rcond=1e-10)
            source = train - basis @ (basis.T @ train)
            truth = target - basis @ (basis.T @ target)
            free = train - hist_basis @ (hist_basis.T @ train)
            histogram = source - free
            if split == 'validation':
                validation_prior = target
                pooled_free, pooled_histogram = free, histogram
            else:
                valid_source = validation_prior - basis @ (basis.T @ validation_prior)
                valid_free = validation_prior - hist_basis @ (hist_basis.T @ validation_prior)
                pooled_free = .5 * (free + valid_free)
                pooled_histogram = .5 * (histogram + valid_source - valid_free)
            if truth @ truth <= 1e-15:
                failures.append(dict(panel=panel, split=split, error='Zero target norm'))
                continue
            rows.append(dict(panel=panel, split=split, family=panel.rsplit('_', 1)[0],
                             free=free, histogram=histogram, truth=truth, pooled_free=pooled_free, pooled_histogram=pooled_histogram))
    def fit(calibration, component):
        predictors = [row[component] if component != 'whole' else row['free'] + row['histogram']
                      for row in calibration]
        target_energy = [row['truth'] @ row['truth'] for row in calibration]
        numerator = sum((x @ row['truth']) / energy for x, row, energy in zip(predictors, calibration, target_energy))
        denominator = sum((x @ x) / energy for x, energy in zip(predictors, target_energy))
        return float(np.clip(numerator / denominator, 0., 1.))

    scores = []
    for row in rows:
        if row['split'] != 'private':
            continue
        calibration = [item for item in rows if item['split'] == 'validation' and item['family'] != row['family']]
        shared = fit(calibration, 'whole')
        separate = (fit(calibration, 'free'), fit(calibration, 'histogram'))
        target = row['truth']
        for source, free, histogram in [
                ('train', row['free'], row['histogram']),
                ('pooled', row['pooled_free'], row['pooled_histogram'])]:
            for method, (free_scale, histogram_scale) in [
                    ('half', (.5, .5)), ('full', (1., 1.)),
                    ('cross_family_shared', (shared, shared)), ('cross_family_separate', separate)]:
                error = free_scale * free + histogram_scale * histogram - target
                scores.append(dict(panel=row['panel'], source=source, method=method,
                                   free_scale=free_scale, histogram_scale=histogram_scale,
                                   relative_l1=np.abs(error).sum() / np.abs(target).sum(),
                                   relative_squared_error=(error @ error) / (target @ target)))
    data = pd.DataFrame(scores)
    data.to_csv(out/'by_panel.csv', index=False)
    data.groupby(['source','method'])[['relative_l1', 'relative_squared_error']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
