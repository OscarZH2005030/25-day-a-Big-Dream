"""Check concentration of zone-total uncertainty under alternative cell variances."""
import msvcrt
import numpy as np
import pandas as pd
from scipy.linalg import orth
import common as d
from audit_direct_od_measurement_design import base

def main():
    out=d.OUT/'task4_od_marginal_spectrum_sensitivity'
    reference=d.OUT/'task4_od_marginal_spectrum'/'by_panel.csv'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__),reference_sha256=d.sha(reference),
        split='private', weighting_split=['train','validation'], powers=[0,.5,1],
        method='Whitened row-space projection; exact count conditioning; pooled prior variance sensitivity.'))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,_,_,h=base.load(panel,'private')
        train_paths,_,_,train,ht=base.load(panel,'train')
        valid_paths,_,_,valid,hv=base.load(panel,'validation')
        assert paths.path_id.equals(train_paths.path_id) and paths.path_id.equals(valid_paths.path_id)
        for source in (h,ht,hv):
            hashes.update(source)
        zones=sorted(set(paths.origin_zone)|set(paths.destination_zone))
        origin=np.array([paths.origin_zone==z for z in zones],dtype=float)
        weights=.5*(train/train.sum()+valid/valid.sum())
        for power in (0,.5,1):
            sd=np.maximum(weights,.1/len(paths))**(power/2)
            weighted=origin*sd
            basis=orth((A*sd).T)
            residual=weighted-(weighted@basis)@basis.T
            values=np.linalg.svd(residual,compute_uv=False)**2
            trace=values.sum()
            rows.append(dict(panel=panel,power=power,zones=len(zones),
                rank=int((values>values[0]*1e-10).sum()),top_one_share=values[0]/trace,
                top_five_share=values[:5].sum()/trace,effective_rank=trace**2/(values@values)))
    frame=pd.DataFrame(rows)
    prior=pd.read_csv(reference)
    pd.testing.assert_frame_equal(frame.query('power==0')[prior.columns].reset_index(drop=True),
                                  prior,atol=1e-10,rtol=0)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby('power')[['top_one_share','top_five_share','effective_rank']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    print(frame.groupby('power').top_five_share.agg(['min','mean','max']))

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
