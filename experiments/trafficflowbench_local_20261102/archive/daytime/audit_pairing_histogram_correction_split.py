"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base





def main():
    out = d.OUT/'task4_pairing_histogram_correction_split'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),predictor_splits=['train','validation'],
        evaluation_splits=['private'],scope='Split the transferred histogram-mask correction into histogram-preserving and histogram-changing components.'))
    rows, hashes, failures=[], {}, []
    for panel in sorted(d.night.predictor.load_routes()):
        train_paths, _, _, train, provenance=base.load(panel,'train')
        hashes.update(provenance)
        train=train/train.sum()
        valid_paths, _, _, valid, provenance=base.load(panel,'validation')
        hashes.update(provenance)
        assert valid_paths.path_id.equals(train_paths.path_id)
        valid=valid/valid.sum()
        for split in ('private',):
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
            for view, constraints in [('margins', np.vstack((counts, groups)))]:
                basis = orth(constraints.T, rcond=1e-10)
                pairs = np.column_stack((train, valid, target))
                source, second, truth = (pairs - basis @ (basis.T @ pairs)).T
                target_l1 = np.abs(truth).sum()
                target_l2 = truth @ truth
                if min(target_l1, target_l2) <= 1e-15:
                    failures.append(dict(panel=panel, view=view, error='Zero residual norm'))
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
                changing = correction - preserving
                assert np.abs(bins @ preserving).max() < 1e-10
                assert abs(preserving @ changing) < 1e-12
                for method, delta in [('uniform', np.zeros_like(average)),
                                      ('full_correction', correction),
                                      ('histogram_preserving', preserving),
                                      ('histogram_changing', changing)]:
                    prediction = average + delta
                    norm = np.linalg.norm(prediction)
                    if norm <= 1e-15:
                        failures.append(dict(panel=panel, view=view, method=method, error='Zero projected prediction'))
                        continue
                    prediction *= np.linalg.norm(average) / norm
                    assert np.abs(constraints@prediction).max()<1e-10
                    error=prediction-truth
                    rows.append(dict(panel=panel,view=view,method=method,
                        correction_l2=float(np.linalg.norm(delta)), relative_l1=np.abs(error).sum()/target_l1,
                        relative_squared_error=(error@error)/target_l2))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['view','method'])[['relative_l1','relative_squared_error']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
