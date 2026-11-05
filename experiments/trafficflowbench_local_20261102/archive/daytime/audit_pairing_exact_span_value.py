"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base




def main():
    out=d.OUT/'task4_pairing_exact_span_value'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),geometry=['validation','private'],
        method='Compare count plus margin geometry with and without hypothetical exact full span histogram. Conditional isotropic OD diagonal from orthogonal projectors. No draws.'))
    rows, hashes=[], {}
    for panel in sorted(d.night.predictor.load_routes()):
        for split in ('validation','private'):
            paths, counts, _, prior, provenance=base.load(panel,split)
            hashes.update(provenance)
            groups=np.vstack([pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                              for key in ('origin_zone','destination_zone')])
            oz,dz=[paths[key].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                   for key in ('origin_zone','destination_zone')]
            span=dz-oz
            spans=np.unique(span)
            bins=(span[:,None]==spans).astype(float)
            matrices=(np.vstack((counts,groups)),np.vstack((counts,groups,bins.T)))
            bases=[orth(matrix.T,rcond=1e-10) for matrix in matrices]
            diagonals=[1.-np.sum(basis**2,axis=1) for basis in bases]
            before,after=diagonals
            assert np.min(after)>=-1e-9 and (after<=before+1e-9).all()
            gained=bases[1].shape[1]-bases[0].shape[1]
            assert gained==len(spans)-2
            weights=prior/prior.sum()
            rows.append(dict(panel=panel,split=split,od_cells=len(paths),span_bins=len(spans),
                baseline_nullity=len(paths)-bases[0].shape[1],remaining_nullity=len(paths)-bases[1].shape[1],
                identified_cells_before=int((before<1e-9).sum()),identified_cells_after=int((after<1e-9).sum()),
                unweighted_variance_reduction=gained/before.sum(),
                prior_weighted_variance_reduction=1-(weights@after)/(weights@before)))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby('split')[['unweighted_variance_reduction','prior_weighted_variance_reduction']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
