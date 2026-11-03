"""Compare deterministic minimum-baseline-movement tie-breaking on the same cases."""
import hashlib
import msvcrt

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_span_shrink_stress import load, metrics


def closest_optimum(base, prior, signed, proposal, origin, dest):
    size=len(base)
    columns=np.arange(size)
    O=sparse.csr_matrix((np.ones(size),(origin,columns)),shape=(origin.max()+1,size))
    D=sparse.csr_matrix((np.ones(size),(dest,columns)),shape=(dest.max()+1,size))
    sign=np.sign(base-prior)
    M=sparse.vstack([O,D,sparse.csr_matrix(sign[None,:])],format='csr')
    zero=sparse.csr_matrix((size,size))
    I=sparse.eye(size,format='csr')
    inequality=sparse.vstack([sparse.hstack([I,-I,zero]),sparse.hstack([-I,-I,zero]),
                              sparse.hstack([I,zero,-I]),sparse.hstack([-I,zero,-I])],format='csr')
    optimum=float(np.abs(proposal-signed).sum())
    equality=sparse.vstack([sparse.hstack([M,sparse.csr_matrix((M.shape[0],2*size))]),
                            sparse.csr_matrix(np.r_[np.zeros(size),np.ones(size),np.zeros(size)][None,:])],format='csr')
    bounds=[(float(b),None) if s>0 else (0.,float(b)) if s<0 else (float(b),float(b))
            for b,s in zip(prior,sign)]+[(0.,None)]*(2*size)
    result=linprog(np.r_[np.zeros(2*size),np.ones(size)],A_ub=inequality,b_ub=np.r_[signed,-signed,base,-base],
                   A_eq=equality,b_eq=np.r_[M@base,optimum],bounds=bounds,method='highs')
    if not result.success:
        return None,result.message
    flow=result.x[:size]
    np.testing.assert_allclose(np.abs(flow-signed).sum(),optimum,rtol=1e-9)
    np.testing.assert_allclose(M@flow,M@base,rtol=1e-10,atol=1e-7)
    assert np.abs(flow-base).sum()<=np.abs(proposal-base).sum()+1e-7
    return flow,''


def main():
    src=d.OUT/'task4_invariant_invisible_shrink/all_cases.csv'
    out=d.OUT/'task4_invariant_shrink_tiebreak'
    d.save(out/'protocol.json',dict(
        scope='Posthoc fixed-objective diagnostic on 40 exposed artificial cases. No tuning or fresh validation.',
        method='Among equal minimum invisible-L1 solutions, minimize total L1 movement from the baseline. Keep all non-OD score constraints fixed.',
        blends=[.05,1.],source_sha256=d.sha(__file__),case_table_sha256=d.sha(src)))
    rows,failures=[],[]
    cases=pd.read_csv(src).query('blend==0')
    for panel,group in cases.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for row in group.itertuples():
            name=hashlib.sha256(row.case.encode()).hexdigest()[:16]+'.npz'
            path=d.WORK/'task4_invariant_invisible_shrink_cases'/name
            assert d.sha(path)==row.artifact_sha256
            with np.load(path) as z:
                base,prior,signed,proposal=[z[k] for k in ['baseline','prior','signed','proposed']]
                closest,message=closest_optimum(base,prior,signed,proposal,origin,dest)
                if closest is None:
                    failures.append(dict(case=row.case,seed=row.seed,message=message))
                    continue
                folder=d.WORK/'task4_invariant_shrink_tiebreak_cases'
                folder.mkdir(parents=True,exist_ok=True)
                artifact=folder/name
                np.savez_compressed(artifact,closest=closest)
                for method,flow in [('default_tie',proposal),('minimum_move_tie',closest)]:
                    for blend in [.05,1.]:
                        prediction=(1.-blend)*base+blend*flow
                        scores=metrics(prediction,z['truth'],prior,A,z['counts'],dest)
                        for key in ['synthetic_S_link','synthetic_S_dev','synthetic_S_attr']:
                            np.testing.assert_allclose(scores[key],getattr(row,key),rtol=0,atol=1e-8)
                        rows.append(dict(case=row.case,panel=panel,noise=row.noise,seed=row.seed,
                                         method=method,blend=blend,artifact_sha256=d.sha(artifact),
                                         relative_od_movement=float(np.abs(prediction-base).sum()/base.sum()),
                                         od_delta=scores['synthetic_S_od']-row.synthetic_S_od))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_case.csv',index=False)
    frame.groupby(['noise','method','blend'])[['relative_od_movement','od_delta']].agg(['count','mean','min','max']).to_csv(out/'summary.csv')
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_invariant_shrink_tiebreak_complete',
        next='Verify tied objectives and exact non-OD invariance before interpreting the posthoc comparison.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
