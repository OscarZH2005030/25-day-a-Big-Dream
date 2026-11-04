"""Quantify how many unresolved zone-total directions dominate covariance."""
import msvcrt
import numpy as np
import pandas as pd
from scipy.linalg import pinvh
import common as d
from audit_direct_od_measurement_design import base

def main():
    out=d.OUT/'task4_od_marginal_spectrum'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__),split='private',
        method='Equal cell variance conditional on exact published counts; spectrum of origin-total covariance.',
        tolerance=1e-10))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,_,_,provenance=base.load(panel,'private')
        hashes.update(provenance)
        zones=sorted(set(paths.origin_zone)|set(paths.destination_zone))
        origin=np.array([paths.origin_zone==z for z in zones],dtype=float)
        destination=np.array([paths.destination_zone==z for z in zones],dtype=float)
        gram_inverse=pinvh(A@A.T)
        projected=origin-(origin@A.T)@gram_inverse@A
        projected_destination=destination-(destination@A.T)@gram_inverse@A
        np.testing.assert_allclose(projected,projected_destination,atol=1e-10,rtol=0)
        values=np.linalg.svd(projected,compute_uv=False)**2
        trace=values.sum()
        assert trace>0
        rows.append(dict(panel=panel,zones=len(zones),rank=int((values>1e-10).sum()),
                         top_one_share=values[0]/trace,top_five_share=values[:5].sum()/trace,
                         effective_rank=trace**2/(values@values)))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    print(frame.describe().to_string())

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
