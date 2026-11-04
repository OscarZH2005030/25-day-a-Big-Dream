"""Stress OD measurement choices across three assumed variance laws."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

import common as d
from audit_direct_od_measurement_design import base

EXPONENTS=(0.,.5,1.,.25,.75)


def main():
    out=d.OUT/'task4_direct_od_variance_interpolation'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__), split='private', budget=16,
        exponents=EXPONENTS, design='Use fixed pooled historical selection weights and private evaluation weights. Select under each law or maximize worst per-law fractional marginal gain relative to the best candidate in that law at that step. Evaluate fixed sets under the three design laws and two interpolation laws excluded from the selection objective.',
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
        for name in ('power_0','power_0.5','power_1','balanced'):
            matrices=[c.copy() for c in starts]
            used=np.zeros(n,dtype=bool)
            for _ in range(16):
                diagonals=[c.diagonal().copy() for c in matrices]
                gains=np.array([np.divide(np.einsum('i,ij,ij->j',w,c,c),var,
                                          out=np.zeros(n),where=var>1e-12)
                                for c,var in zip(matrices,diagonals)])
                eligible=(~used)&np.all(np.array(diagonals[:3])>1e-12,axis=0)
                gains[:,~eligible]=0
                assert eligible.any()
                if name=='balanced':
                    objective=(gains[:3]/gains[:3].max(axis=1,keepdims=True)).min(axis=0)
                else:
                    objective=gains[EXPONENTS.index(float(name[6:]))]
                objective[~eligible]=-np.inf
                j=int(np.argmax(objective))
                used[j]=True
                for c,var in zip(matrices,diagonals):
                    column=c[:,j].copy()
                    if var[j]>1e-12:
                        c-=np.outer(column,column)/var[j]
            for power,start,c in zip(EXPONENTS,starts,matrices):
                before=start.diagonal()
                reduction=evaluation*(before-c.diagonal())
                scale=evaluation@before
                rows.append(dict(panel=panel,design=name,exponent=power,
                                 total_reduction=reduction.sum()/scale,unselected_reduction=reduction[~used].sum()/scale))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby(['design','exponent'])[['total_reduction','unselected_reduction']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='task4_direct_od_variance_interpolation_complete',
            next='Verify original grid matches and compare interpolation regret.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
