"""Measure repeatability of the unobserved pairing component of the released priors."""
import msvcrt

import numpy as np
import pandas as pd
from scipy.linalg import null_space

import common as d
from audit_direct_od_measurement_design import base






def main():
    out=d.OUT/'task4_pairing_reversible_dimension'
    baseline=getattr(d,'FROZEN_TASK4_BASELINE',None)
    if baseline is None:
        baseline=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(source_sha256=d.sha(__file__),baseline_sha256=d.sha(baseline),
        evaluation_split='private',
        method='Compute the local two-sided OD nullspace fixing zero flows, exact prior ties, counts, margins, prior L1, and optionally the span histogram. No draws.'))
    frozen=pd.read_csv(baseline,dtype={'panel':str,'departure_time':str,'path_id':str})
    rows,hashes,failures=[],{},[]
    for panel in sorted(d.night.predictor.load_routes()):
        paths,counts,_,prior,provenance=base.load(panel,'private')
        hashes.update(provenance)
        departure=pd.read_csv(d.DATA/'task4'/panel/'private'/'synthetic_weak_prior.csv',dtype=str).departure_time.iloc[0]
        start=frozen.query('panel==@panel and departure_time==@departure').set_index('path_id').reindex(paths.path_id).path_flow.to_numpy()
        assert np.isfinite(start).all() and start.min()>=0
        active=(start>1e-9)&(np.abs(start-prior)>1e-9)
        signs=np.sign(start-prior)
        groups=np.vstack([pd.get_dummies(paths[k]).to_numpy(dtype=float).T for k in ('origin_zone','destination_zone')])
        origin,destination=[paths[k].str.extract(r'^Z([0-9]+)$',expand=False).astype(int).to_numpy()
                            for k in ('origin_zone','destination_zone')]
        span=destination-origin
        bins=(span[:,None]==np.unique(span)).astype(float).T
        for view,matrix in [('margins',np.vstack((counts,groups,signs))),
                            ('margins_and_histogram',np.vstack((counts,groups,bins,signs)))]:
            subspace=null_space(matrix[:,active],rcond=1e-10)
            dimension=subspace.shape[1]
            assert dimension>0
            projection_diagonal=np.sum(subspace**2,axis=1)
            coordinate=int(np.argmax(projection_diagonal))
            direction=np.zeros(len(start))
            direction[active]=subspace@subspace[coordinate]
            slack=np.minimum(start[active],np.abs(start[active]-prior[active]))
            changed=np.abs(direction[active])>1e-12
            scale=.5*np.min(slack[changed]/np.abs(direction[active][changed]))
            delta=scale*direction
            assert np.abs(matrix@delta).max()<1e-7
            for flow in (start+delta,start-delta):
                assert flow.min()>=-1e-9
                assert abs(np.abs(flow-prior).sum()-np.abs(start-prior).sum())<1e-7
            rows.append(dict(panel=panel,view=view,od_cells=len(start),active_cells=int(active.sum()),
                reversible_dimension=dimension,reversible_cells=int((projection_diagonal>1e-10).sum()),
                witness_l1_relative=np.abs(delta).sum()/start.sum(),max_constraint_drift=np.abs(matrix@delta).max()))
    data=pd.DataFrame(rows)
    data.to_csv(out/'by_panel.csv',index=False)
    data.groupby('view')[['active_cells','reversible_dimension','reversible_cells']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
