"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base



def main():
    out = d.OUT/'task4_pairing_covariance_aliases'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__), geometry='validation', weights='train',
        method='Project OD coordinate vectors into the count-plus-margin nullspace. Choose extremes of weighted variance for equal unit trace additions to a shared-factor covariance. No random draws.'))
    rows, hashes = [], {}
    for panel in sorted(d.night.predictor.load_routes()):
        paths, counts, _, _, hv = base.load(panel,'validation')
        tpaths, _, _, train_prior, ht = base.load(panel,'train')
        assert paths.path_id.equals(tpaths.path_id)
        hashes.update(hv)
        hashes.update(ht)
        weights = train_prior/train_prior.sum()
        groups = np.vstack([pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                            for key in ('origin_zone','destination_zone')])
        measurements = np.vstack((counts, groups))
        row_space = orth(measurements.T, rcond=1e-10)
        projector = np.eye(len(paths)) - row_space @ row_space.T
        norm_squared = projector.diagonal()
        eligible = np.flatnonzero(norm_squared > 1e-9)
        weighted = (weights @ (projector[:,eligible]**2)) / norm_squared[eligible]
        selected = {'minimum': eligible[weighted.argmin()], 'maximum': eligible[weighted.argmax()]}
        factor = groups.T / np.sqrt(2)
        np.testing.assert_allclose(projector @ factor,0,atol=1e-9,rtol=0)
        prior_trace = np.sum(factor**2)
        for method, i in selected.items():
            direction = projector[:,i]/np.sqrt(norm_squared[i])
            np.testing.assert_allclose(np.linalg.norm(direction),1,atol=1e-9,rtol=0)
            maximum_drift = np.abs(measurements @ direction).max()
            assert maximum_drift < 1e-9
            rows.append(dict(panel=panel,method=method,od_cells=len(paths),
                measurement_rank=row_space.shape[1], pairing_nullity=len(paths)-row_space.shape[1],
                unweighted_added_trace=prior_trace,
                weighted_pairing_trace=prior_trace*(weights @ direction**2),
                max_observable_factor_drift=maximum_drift))
    data = pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    wide=data.pivot(index='panel',columns='method',values='weighted_pairing_trace')
    wide['ratio']=wide.maximum/wide.minimum
    wide.to_csv(out/'contrast.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
