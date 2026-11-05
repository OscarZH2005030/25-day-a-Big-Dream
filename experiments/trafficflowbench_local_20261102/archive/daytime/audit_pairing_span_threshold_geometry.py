"""Construct covariance aliases invisible to released counts and exact margins."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth

import common as d
from audit_direct_od_measurement_design import base




def main():
    out=d.OUT/'task4_pairing_span_threshold_geometry'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),geometry='validation',
        method='Project all nonconstant integer long-trip threshold queries and a linear span control onto the count and count-plus-margin nullspaces. No draws.'))
    rows, hashes=[], {}
    for panel in sorted(d.night.predictor.load_routes()):
        paths, counts, _, _, provenance=base.load(panel,'validation')
        hashes.update(provenance)
        groups=np.vstack([pd.get_dummies(paths[key]).to_numpy(dtype=float).T
                          for key in ('origin_zone','destination_zone')])
        oz,dz=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
               for k in ('origin_zone','destination_zone')]
        span=dz-oz
        thresholds=np.arange(span.min(),span.max())
        queries=np.column_stack((span,span[:,None]>thresholds))
        full_norm=np.sum(queries**2,axis=0)
        ratios=[]
        for matrix in (counts,np.vstack((counts,groups))):
            basis=orth(matrix.T,rcond=1e-10)
            residual=queries-basis@(basis.T@queries)
            ratios.append(np.sum(residual**2,axis=0)/full_norm)
        assert (ratios[1]<=ratios[0]+1e-12).all() and ratios[1][0]<1e-18
        for i,threshold in enumerate(np.r_[-1,thresholds]):
            rows.append(dict(panel=panel,query='total_span' if i==0 else 'long_trip',
                threshold=int(threshold),count_null_share=ratios[0][i],joint_null_share=ratios[1][i]))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.query("query=='long_trip'").groupby('panel')[['count_null_share','joint_null_share']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
