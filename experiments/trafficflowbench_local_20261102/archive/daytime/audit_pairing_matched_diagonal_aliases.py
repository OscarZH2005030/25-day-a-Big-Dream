"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base



def main():
    out = d.OUT/'task4_pairing_matched_diagonal_aliases'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__), geometry='validation', weights='train',
        method='Construct disjoint unit null directions by projecting prior separately on alternating prior-ranked OD subsets. Sum and difference match all cell variances. Compare the prior linear-functional variance. No draws.'))
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
        order = np.argsort(weights,kind='stable')
        directions = []
        for subset in (order[::2],order[1::2]):
            row_space = orth(measurements[:,subset].T,rcond=1e-10)
            projected = weights[subset] - row_space @ (row_space.T @ weights[subset])
            assert np.linalg.norm(projected) > 1e-9
            direction = np.zeros(len(paths))
            direction[subset] = projected/np.linalg.norm(projected)
            directions.append(direction)
        left,right = directions
        positive = (left+right)/np.sqrt(2)
        negative = (left-right)/np.sqrt(2)
        np.testing.assert_allclose(positive**2,negative**2,atol=1e-15,rtol=0)
        drift = max(np.abs(measurements @ factor).max() for factor in (positive,negative))
        assert drift < 1e-9
        for law,factor in [('sum',positive),('difference',negative)]:
            rows.append(dict(panel=panel,law=law,unit_trace=factor @ factor,
                weighted_pairing_trace=weights @ factor**2,prior_functional_variance=(weights @ factor)**2,
                max_observable_factor_drift=drift))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    wide=data.pivot(index='panel',columns='law',values='prior_functional_variance')
    wide['ratio']=wide['sum']/wide['difference']
    wide.to_csv(out/'contrast.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
