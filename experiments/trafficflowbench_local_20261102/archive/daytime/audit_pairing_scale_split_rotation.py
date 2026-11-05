"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_scale_split_rotation'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),predictor_splits='two non-target released splits',geometry_split='private',
        evaluation_splits=['train','validation','private'],scope='Rotate the target split using one fixed scale and a common count-plus-margins geometry per road.'))
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        split_data = {}
        for split in ('train', 'validation', 'private'):
            paths, counts, _, prior, provenance = base.load(panel, split)
            hashes.update(provenance)
            split_data[split] = paths, counts, prior / prior.sum()
        paths, counts, _ = split_data['private']
        assert all(paths.path_id.equals(item[0].path_id) for item in split_data.values())
        for split, (_, _, target) in split_data.items():
            train, valid = [item[2] for name, item in split_data.items() if name != split]
            groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T
                              for k in ('origin_zone','destination_zone')])
            origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                                for k in ('origin_zone','destination_zone')]
            span=destination-origin
            bins=(span[:,None]==np.unique(span)).astype(float).T
            for view, constraints in [('margins', np.vstack((counts, groups)))]:
                basis = orth(constraints.T, rcond=1e-10)
                pairs = np.column_stack((train, valid, target))
                source, second, truth = (pairs - basis @ (basis.T @ pairs)).T
                target_l1 = np.abs(truth).sum()
                target_l2 = truth @ truth
                if min(target_l1, target_l2) <= 1e-15:
                    failures.append(dict(panel=panel, split=split, view=view, error='Zero residual norm'))
                    continue
                average = .25 * (source + second)
                histogram_basis = orth(np.vstack((constraints, bins)).T, rcond=1e-10)
                hist_source, hist_second = (pairs[:, :2] - histogram_basis @ (histogram_basis.T @ pairs[:, :2])).T
                magnitude = np.abs(hist_source) + np.abs(hist_second)
                disagreement = np.divide(np.abs(hist_source - hist_second), magnitude,
                                         out=np.zeros_like(magnitude), where=magnitude > 1e-12)
                mask = (np.signbit(hist_source) == np.signbit(hist_second)) & (disagreement >= .5)
                correction = np.where(mask, -.3 * average, 0.)
                correction -= basis @ (basis.T @ correction)
                preserving = correction - histogram_basis @ (histogram_basis.T @ correction)
                assert np.abs(bins @ preserving).max() < 1e-10
                free = average - histogram_basis @ (histogram_basis.T @ average)
                fixed = average - free
                updated_free = free + preserving
                if min(np.linalg.norm(updated_free), np.linalg.norm(average + preserving)) <= 1e-15:
                    failures.append(dict(panel=panel, split=split, view=view, error='Zero updated component'))
                    continue
                scale = np.linalg.norm(average) / np.linalg.norm(average + preserving)
                candidates = [
                    ('uniform', average),
                    ('fixed_whole', 1.03 * average),
                    ('fixed_histogram', free + 1.03 * fixed),
                    ('fixed_free', 1.03 * free + fixed),
                    ('mask_scale_only', scale * average),
                ]
                for method, prediction in candidates:
                    assert np.abs(constraints @ prediction).max() < 1e-10
                    histogram_delta = float(np.abs(bins @ (prediction - average)).max())
                    error=prediction-truth
                    rows.append(dict(panel=panel,split=split,view=view,method=method,
                        scale=float(scale), histogram_delta=histogram_delta, correction_l2=float(np.linalg.norm(prediction - average)), relative_l1=np.abs(error).sum()/target_l1,
                        relative_squared_error=(error@error)/target_l2))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['split','view','method'])[['relative_l1','relative_squared_error']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
