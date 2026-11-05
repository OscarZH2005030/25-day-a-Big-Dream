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
    out=d.OUT/'task4_pairing_symmetric_ablation'
    baseline=getattr(d,'FROZEN_TASK4_BASELINE',None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        evaluation_split='private',
        method='Compare positivity alone, fixed prior signs, and fixed signs plus exact prior L1 at the symmetric target-aligned optimum. No draws.'))
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
            positivity=np.minimum(start,np.abs(target_gap))
            fixed_sign=np.minimum(positivity,np.abs(start-prior))
            last_radius=float('inf')
            for regime,limits,constraints in [('positivity',positivity,matrix[:-1]),
                                               ('fixed_sign',fixed_sign,matrix[:-1]),
                                               ('exact_l1',fixed_sign,matrix)]:
                equality=sparse.csr_matrix(constraints*direction)
                result=linprog(-np.ones(len(start)),A_eq=equality,b_eq=np.zeros(constraints.shape[0]),
                    bounds=list(zip(np.zeros(len(start)),limits)),method='highs')
                if not result.success:
                    failures.append(dict(panel=panel,view=view,regime=regime,error=result.message))
                    continue
                delta=result.x*direction
                drift=np.abs(constraints@delta).max()
                assert drift<1e-7
                prior_deviations=[]
                for flow in (start+delta,start-delta):
                    assert flow.min()>=-1e-9
                    if regime!='positivity':
                        assert ((flow-prior)*signs>=-1e-8).all()
                    prior_deviations.append((np.abs(flow-prior).sum()-np.abs(start-prior).sum())/start.sum())
                if regime=='exact_l1':
                    assert max(map(abs,prior_deviations))<1e-9
                move=np.abs(delta).sum()/start.sum()
                assert move<=last_radius+1e-9
                last_radius=move
                rows.append(dict(panel=panel,view=view,regime=regime,symmetric_l1_radius=move,
                    maximum_prior_distance_change=max(map(abs,prior_deviations)),max_constraint_drift=drift))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby(['view','regime']).symmetric_l1_radius.agg(['min','mean','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
