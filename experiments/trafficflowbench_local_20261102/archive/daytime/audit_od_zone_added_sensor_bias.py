"""Check concentration of zone-total uncertainty under alternative cell variances."""
import msvcrt
import numpy as np
import pandas as pd
from scipy.linalg import orth
import common as d
from audit_direct_od_measurement_design import base

def main():
    out=d.OUT/'task4_od_zone_added_sensor_bias'
    reference=d.OUT/'task4_od_direct_zone_total_bound'/'by_panel.csv'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__),reference_sha256=d.sha(reference),
        split='private', weighting_split=['train','validation'], powers=[0,.5,1],
        method='Whitened row-space projection; exact count conditioning; pooled prior variance sensitivity; Select 16 exact origin totals under each power and evaluate frozen selections under all three powers. Also choose a balanced set maximizing the worst marginal gain as a fraction of initial trace across the three laws. Score fixed sets with common sensor bias variance equal to independent noise variance, holding independent variance fixed at 0.25 and 1 times mean eligible origin variance. Resolve near ties by sorted zone order within 1e-12 of initial trace.'))
    rows,hashes=[],{}
    transfer=[]
    noisy=[]
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
        selections,initial_covariances={},{}
        for power in (0,.5,1):
            sd=np.maximum(weights,.1/len(paths))**(power/2)
            weighted=origin*sd
            basis=orth((A*sd).T)
            residual=weighted-(weighted@basis)@basis.T
            values=np.linalg.svd(residual,compute_uv=False)**2
            trace=values.sum()
            covariance=residual@residual.T
            initial_covariances[power]=covariance.copy()
            ids=[]
            for _ in range(16):
                variance=covariance.diagonal()
                gain=np.divide(np.sum(covariance**2,axis=0),variance,out=np.zeros_like(variance),where=variance>trace*1e-12)
                j=int(np.flatnonzero(gain >= gain.max()-trace*1e-12)[0])
                assert gain[j]>0 and j not in ids
                ids.append(j)
                column=covariance[:,j].copy()
                covariance-=np.outer(column,column)/variance[j]
            selections[power]=ids
            zone_gain=1-covariance.trace()/trace
            assert 0<=zone_gain<=values[:16].sum()/trace+1e-12
            rows.append(dict(panel=panel,power=power,zones=len(zones),
                rank=int((values>values[0]*1e-10).sum()),top_one_share=values[0]/trace,
                top_five_share=values[:5].sum()/trace,effective_rank=trace**2/(values@values), zone_gain_16=zone_gain, upper_bound_16=values[:16].sum()/trace, probes_for_half=int(np.searchsorted(np.cumsum(values),trace/2)+1)))
        current=[matrix.copy() for matrix in initial_covariances.values()]
        traces=[matrix.trace() for matrix in current]
        selected=[]
        for _ in range(16):
            gains=[]
            for matrix,initial_trace in zip(current,traces):
                variance=matrix.diagonal()
                gains.append(np.divide(np.sum(matrix**2,axis=0),variance*initial_trace,
                    out=np.zeros_like(variance),where=variance>initial_trace*1e-12))
            score=np.min(gains,axis=0)
            score[selected]=-np.inf
            j=int(np.flatnonzero(score>=score.max()-1e-12)[0])
            assert score[j]>0
            selected.append(j)
            for matrix in current:
                column=matrix[:,j].copy()
                matrix-=np.outer(column,column)/column[j]
        selections['balanced']=selected
        for selected_power,ids in selections.items():
            for evaluated_power,initial in initial_covariances.items():
                cross=initial[:,ids]
                gain=np.sum((cross@np.linalg.pinv(initial[np.ix_(ids,ids)]))*cross)/initial.trace()
                if selected_power==evaluated_power:
                    expected=next(row["zone_gain_16"] for row in rows[-3:] if row["power"]==evaluated_power)
                    np.testing.assert_allclose(gain,expected,atol=1e-10,rtol=0)
                transfer.append(dict(panel=panel,selected_power=selected_power,evaluated_power=evaluated_power,gain=gain))
                for noise_ratio in (.25,1.):
                    block=initial[np.ix_(ids,ids)]
                    noise=noise_ratio*initial.trace()/(len(initial)-2)
                    noisy_gain=np.sum((cross@np.linalg.pinv(block+noise*(np.eye(len(ids))+np.ones((len(ids),len(ids))))))*cross)/initial.trace()
                    assert 0<=noisy_gain<=gain+1e-12
                    noisy.append(dict(panel=panel,selected_power=selected_power,evaluated_power=evaluated_power,noise_ratio=noise_ratio,gain=noisy_gain))
    pd.DataFrame(noisy).to_csv(out/'noisy.csv',index=False)
    frame=pd.DataFrame(rows)
    prior=pd.read_csv(reference)
    pd.testing.assert_frame_equal(frame[prior.columns],
                                  prior,atol=1e-10,rtol=0)
    frame.to_csv(out/'by_panel.csv',index=False)
    pd.DataFrame(transfer).to_csv(out/'transfer.csv',index=False)
    print(pd.DataFrame(transfer).groupby(['selected_power','evaluated_power']).gain.mean().unstack())
    frame.groupby('power')[['top_one_share','top_five_share','effective_rank','zone_gain_16','upper_bound_16','probes_for_half']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    print(frame.groupby('power')[['zone_gain_16','upper_bound_16','probes_for_half']].agg(['min','mean','max']))

if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
