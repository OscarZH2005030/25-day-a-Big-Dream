"""Compare balanced designs on the same unmeasured OD cells."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

import common as d
from audit_direct_od_measurement_design import base

EXPONENTS=(0.,.5,1.,.25,.75)


def main():
    out=d.OUT/'task4_direct_od_fixed_noise'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__), split='private', budget=16,
        exponents=EXPONENTS, design='Freeze exact-measurement selections. Score both on their shared unmeasured cells at noise variances 0, 0.25 and 1 using batch covariance conditioning.',
        scope='Hypothetical direct measurements. Variance laws are sensitivity assumptions, not fitted to hidden OD errors.'))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,_,private,h=base.load(panel,'private')
        t,_,_,train,ht=base.load(panel,'train')
        v,_,_,valid,hv=base.load(panel,'validation')
        assert paths.path_id.equals(t.path_id) and paths.path_id.equals(v.path_id)
        for provenance in (h,ht,hv):
            hashes.update(provenance)
        w=.5*(train/train.sum()+valid/valid.sum())
        evaluation=private/private.sum()
        n=len(paths)
        starts=[]
        for power in EXPONENTS:
            variance=np.maximum(w,.1/n)**power
            variance/=variance.mean()
            cross=A*variance
            covariance=np.diag(variance)-cross.T@pinvh(cross@A.T)@cross
            np.testing.assert_allclose(A@covariance,0,atol=1e-8,rtol=0)
            starts.append(covariance)
        outcomes={}
        for name in ('total','other'):
            matrices=[c.copy() for c in starts]
            used=np.zeros(n,dtype=bool)
            for _ in range(16):
                diagonals=[c.diagonal().copy() for c in matrices]
                gains=np.array([np.divide(np.einsum('i,ij,ij->j',w,c,c),var,
                                          out=np.zeros(n),where=var>1e-12)
                                for c,var in zip(matrices[:3],diagonals[:3])])
                eligible=(~used)&np.all(np.array(diagonals[:3])>1e-12,axis=0)
                if name=="other":
                    gains-=w*np.array(diagonals[:3])
                gains=np.maximum(gains,0.)
                gains[:,~eligible]=0
                assert eligible.any()
                objective=(gains/np.maximum(gains.max(axis=1,keepdims=True),1e-30)).min(axis=0)
                objective[~eligible]=-np.inf
                j=int(np.argmax(objective))
                used[j]=True
                for c,var in zip(matrices,diagonals):
                    column=c[:,j].copy()
                    if var[j]>1e-12:
                        c-=np.outer(column,column)/var[j]
            outcomes[name]=(used,matrices)
        common=~(outcomes["total"][0] | outcomes["other"][0])
        for name,(used,matrices) in outcomes.items():
            for power,start in zip(EXPONENTS,starts):
                cross=start[:,used]
                block=start[np.ix_(used,used)]
                scale=evaluation@start.diagonal()
                for noise in (0.,.25,1.):
                    gain=np.einsum('ij,ij->i',cross@pinvh(block+noise*np.eye(block.shape[0])),cross)
                    reduction=evaluation*gain/scale
                    rows.append(dict(panel=panel,design=name,exponent=power,noise=noise,
                                     total_reduction=reduction.sum(),common_reduction=reduction[common].sum()))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby(['design','exponent','noise'])[['total_reduction','common_reduction']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='task4_direct_od_fixed_noise_complete',
            next='Compare gains with total-target selections using matched roads and variance laws.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
