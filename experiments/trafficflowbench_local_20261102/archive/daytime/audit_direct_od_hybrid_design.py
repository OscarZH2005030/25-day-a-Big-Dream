"""Compare direct selection against a rule targeting other OD cells."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

import common as d
from audit_direct_od_measurement_design import base


def main():
    out=d.OUT/'task4_direct_od_hybrid_design'
    d.save(out/'protocol.json',dict(code_sha256=d.sha(__file__), split='private', budget=16,
        design='Preallocate 0, 4, 8, 12 or 16 selections to total weighted gain, then spend the remainder on unselected-cell gain. Use the fixed pooled historical weights and evaluate all designs with private weights.',
        scope='Hypothetical exact measurements on released count geometry, isotropic diagnostic covariance.'))
    rows,hashes=[],{}
    for panel in d.night.predictor.load_routes():
        paths,A,_,private,h=base.load(panel,'private')
        t,_,_,train,ht=base.load(panel,'train')
        v,_,_,valid,hv=base.load(panel,'validation')
        assert paths.path_id.equals(t.path_id) and paths.path_id.equals(v.path_id)
        for digest in (h,ht,hv):
            hashes.update(digest)
        w=.5*(train/train.sum()+valid/valid.sum())
        evaluation=private/private.sum()
        n=len(paths)
        initial=np.eye(n)-A.T@pinvh(A@A.T)@A
        baseline=initial.diagonal().copy()
        for direct_steps in (0,4,8,12,16):
            current=initial.copy()
            used=np.zeros(n,dtype=bool)
            for step in range(16):
                diagonal=current.diagonal().copy()
                gain=np.divide(np.einsum('i,ij,ij->j',w,current,current),diagonal,
                               out=np.zeros(n),where=diagonal>1e-12)
                if step>=direct_steps:
                    gain-=w*diagonal
                gain[used | (diagonal<=1e-12)]=-np.inf
                index=int(np.argmax(gain))
                used[index]=True
                column=current[:,index].copy()
                current-=np.outer(column,column)/diagonal[index]
            reduction=evaluation*(baseline-current.diagonal())
            scale=evaluation@baseline
            rows.append(dict(panel=panel,direct_steps=direct_steps,
                             total_reduction=reduction.sum()/scale,
                             unselected_reduction=reduction[~used].sum()/scale,
                             selected_reduction=reduction[used].sum()/scale))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby('direct_steps')[['total_reduction','unselected_reduction','selected_reduction']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='task4_direct_od_hybrid_design_complete',
            next='Independently verify decomposition and comparison with the original pooled design.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
