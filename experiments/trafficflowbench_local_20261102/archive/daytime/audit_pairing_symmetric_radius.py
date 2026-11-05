"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import orth
from scipy import sparse
from scipy.optimize import linprog

import common as d
from audit_direct_od_measurement_design import base






def main():
    out=d.OUT/'task4_pairing_symmetric_radius'
    baseline=getattr(d,'FROZEN_TASK4_BASELINE',None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        evaluation_split='private',
        method='Maximize symmetric feasible L1 movement aligned cellwise with the signed two-source pairing target. No draws.'))
    frozen=pd.read_csv(baseline,dtype={'panel':str,'departure_time':str,'path_id':str})
    rows,hashes,failures=[],{},[]
    for panel in sorted(d.night.predictor.load_routes()):
        paths,counts,_,prior,provenance=base.load(panel,'private')
        hashes.update(provenance)
        sources=[]
        for split in ('train','validation'):
            source_paths,_,_,source,provenance=base.load(panel,split)
            assert paths.path_id.equals(source_paths.path_id)
            hashes.update(provenance)
            sources.append(source/source.sum())
        departure=pd.read_csv(d.DATA/'task4'/panel/'private'/'synthetic_weak_prior.csv',dtype=str).departure_time.iloc[0]
        start=frozen.query('panel==@panel and departure_time==@departure').set_index('path_id').reindex(paths.path_id).path_flow.to_numpy()
        assert np.isfinite(start).all() and start.min()>=0
        target=.5*np.mean(sources,axis=0)*start.sum()
        displacement=target-start
        signs=np.sign(start-prior)
        groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T for k in ('origin_zone','destination_zone')])
        origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                            for k in ('origin_zone','destination_zone')]
        span=destination-origin
        bins=(span[:,None]==np.unique(span)).astype(float).T
        for view,matrix in [('margins',np.vstack((counts,groups,signs))),
                            ('margins_and_histogram',np.vstack((counts,groups,bins,signs)))]:
            row_basis=orth(matrix[:-1].T,rcond=1e-10)
            target_gap=displacement-row_basis@(row_basis.T@displacement)
            direction=np.sign(target_gap)
            limits=np.minimum(np.minimum(start,np.abs(start-prior)),np.abs(target_gap))
            equality=sparse.csr_matrix(matrix*direction)
            result=linprog(-np.ones(len(start)),A_eq=equality,b_eq=np.zeros(matrix.shape[0]),
                bounds=list(zip(np.zeros(len(start)),limits)),method='highs')
            if not result.success:
                failures.append(dict(panel=panel,view=view,error=result.message))
                continue
            delta=result.x*direction
            drift=np.abs(matrix@delta).max()
            assert drift<1e-7
            for flow in (start+delta,start-delta):
                assert flow.min()>=-1e-9
                assert abs(np.abs(flow-prior).sum()-np.abs(start-prior).sum())<1e-7
            total=start.sum()
            move=np.abs(delta).sum()/total
            proxy_error=np.abs(target_gap).sum()
            gain=(proxy_error-np.abs(target_gap-delta).sum())/total
            opposite_loss=(np.abs(target_gap+delta).sum()-proxy_error)/total
            np.testing.assert_allclose([gain,opposite_loss],move,atol=1e-9,rtol=0)
            rows.append(dict(panel=panel,view=view,symmetric_l1_radius=move,proxy_gain=gain,
                opposite_proxy_loss=opposite_loss,max_constraint_drift=drift))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby('view').symmetric_l1_radius.agg(['min','mean','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
