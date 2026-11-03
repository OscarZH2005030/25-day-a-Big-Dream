"""Audit why count-only greedy OD selection coincides with largest-prior selection."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import pinvh

import common as d
from audit_direct_od_measurement_design import base


def main():
    out = d.OUT/'task4_direct_od_count_geometry'
    d.save(out/'protocol.json', dict(code_sha256=d.sha(__file__), budget=16, split='private',
        design='Use the count-row Gram pseudoinverse, independently of the original QR basis. Record exact selection agreement and the fraction of each greedy gain attributable to uncertainty in other OD cells.',
        scope='Hypothetical exact direct OD measurements on released count geometry. No hidden labels.'))
    rows, hashes = [], {}
    for panel in d.night.predictor.load_routes():
        paths,A,_,private,h = base.load(panel,'private')
        _,_,train,train_prior,ht = base.load(panel,'train')
        _,_,valid,valid_prior,hv = base.load(panel,'validation')
        hashes.update(h)
        hashes.update(ht)
        hashes.update(hv)
        n=len(paths)
        covariance=np.eye(n)-A.T@pinvh(A@A.T)@A
        np.testing.assert_allclose(A@covariance,0,atol=1e-8)
        weights_by_name = {'train':train_prior/train_prior.sum(),'validation':valid_prior/valid_prior.sum(),
                           'private':private/private.sum()}
        weights_by_name['pooled']=(weights_by_name['train']+weights_by_name['validation'])/2
        for label,w in weights_by_name.items():
            top=np.argsort(-w,kind='stable')[:16]
            current=covariance.copy()
            chosen=[]
            direct, total = 0., 0.
            for step in range(16):
                diagonal=current.diagonal().copy()
                gain=np.divide(np.einsum('i,ij,ij->j',w,current,current),diagonal,
                               out=np.zeros(n),where=diagonal>1e-12)
                gain[chosen]=-1
                index=int(np.argmax(gain))
                direct+=w[index]*diagonal[index]
                total+=gain[index]
                chosen.append(index)
                column=current[:,index].copy()
                current-=np.outer(column,column)/diagonal[index]
            weighted=weights_by_name['private']
            rows.append(dict(panel=panel,weight_source=label,shared_cells=len(set(chosen)&set(top)),
                             identical_order=bool(np.array_equal(chosen,top)),
                             off_cell_gain_share=1-direct/total,
                             remaining_private_trace_ratio=(weighted@current.diagonal())/(weighted@covariance.diagonal())))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby('weight_source')[['shared_cells','identical_order','off_cell_gain_share']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',[])
    d.state(status='stage_complete',stage='task4_direct_od_count_geometry_complete',
            next='Check independent counts-only selection agreement and off-cell gain.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
