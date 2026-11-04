"""Check concentration of zone-total uncertainty under alternative cell variances."""
import msvcrt
import numpy as np
import pandas as pd
from scipy.linalg import orth
import common as d
from audit_direct_od_measurement_design import base

def main():
    out=d.OUT/'task4_od_direct_zone_total_bound'
    reference=d.OUT/'task4_od_marginal_rank_bound'/'by_panel.csv'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__),reference_sha256=d.sha(reference),
        split='private', weighting_split=['train','validation'], powers=[0,.5,1],
        method='Whitened row-space projection; exact count conditioning; pooled prior variance sensitivity; Compare 16 greedy exact origin-total measurements with the arbitrary linear measurement bound.'))
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
            covariance=residual@residual.T
            for _ in range(16):
                variance=covariance.diagonal()
                gain=np.divide(np.sum(covariance**2,axis=0),variance,out=np.zeros_like(variance),where=variance>trace*1e-12)
                j=int(np.argmax(gain))
                assert gain[j]>0
                column=covariance[:,j].copy()
                covariance-=np.outer(column,column)/variance[j]
            zone_gain=1-covariance.trace()/trace
            assert 0<=zone_gain<=values[:16].sum()/trace+1e-12
            rows.append(dict(panel=panel,power=power,zones=len(zones),
                rank=int((values>values[0]*1e-10).sum()),top_one_share=values[0]/trace,
                top_five_share=values[:5].sum()/trace,effective_rank=trace**2/(values@values), zone_gain_16=zone_gain, upper_bound_16=values[:16].sum()/trace, probes_for_half=int(np.searchsorted(np.cumsum(values),trace/2)+1)))
    frame=pd.DataFrame(rows)
    prior=pd.read_csv(reference)
    pd.testing.assert_frame_equal(frame[prior.columns],
                                  prior,atol=1e-10,rtol=0)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby('power')[['top_one_share','top_five_share','effective_rank','zone_gain_16','upper_bound_16','probes_for_half']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    print(frame.groupby('power')[['zone_gain_16','upper_bound_16','probes_for_half']].agg(['min','mean','max']))

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
