"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base




def main():
    out=d.OUT/'task4_pairing_span_histogram_geometry'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),geometry='validation',
        method='Audit the projected span histogram rank under counts and under counts with exact margins. No draws.'))
    rows, hashes=[], {}
    for panel in sorted(d.night.predictor.load_routes()):
        paths, counts, _, _, provenance=base.load(panel,'validation')
        hashes.update(provenance)
        groups=np.vstack([pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                          for key in ('origin_zone','destination_zone')])
        oz,dz=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
               for k in ('origin_zone','destination_zone')]
        span=dz-oz
        spans=np.unique(span)
        bins=(span[:,None]==spans).astype(float)
        for view,matrix in [('counts',counts),('counts_and_margins',np.vstack((counts,groups)))]:
            basis=orth(matrix.T,rcond=1e-10)
            residual=bins-basis@(basis.T@bins)
            values=np.linalg.svd(residual,compute_uv=False)
            rank=int((values>values[0]*1e-10).sum())
            rows.append(dict(panel=panel,view=view,bins=len(spans),unidentified_rank=rank,
                identified_rank=len(spans)-rank,
                total_mass_relative=np.linalg.norm(residual.sum(axis=1))/np.sqrt(len(span)),
                total_span_relative=np.linalg.norm(residual@spans)/np.linalg.norm(span)))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
