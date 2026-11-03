"""Compare a per-path cap with the same-movement global-budget direction."""
import hashlib
import msvcrt

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_budgeted_source_direction import BUDGET
from task4_span_shrink_stress import load
from verify_cycle_budget import score

PATH_CAP=.05


def capped_solution(base,prior,origin,dest,direction):
    n=len(base)
    columns=np.arange(n)
    O=sparse.csr_matrix((np.ones(n),(origin,columns)),shape=(origin.max()+1,n))
    D=sparse.csr_matrix((np.ones(n),(dest,columns)),shape=(dest.max()+1,n))
    sign=np.sign(base-prior)
    M=sparse.vstack([O,D,sparse.csr_matrix(sign[None,:])],format='csr')
    equality=sparse.hstack([M,sparse.csr_matrix(M.shape)],format='csr')
    I=sparse.eye(n,format='csr')
    inequality=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I]),
                              sparse.csr_matrix(np.r_[np.zeros(n),np.ones(n)][None,:])],format='csr')
    lower=np.maximum(np.where(sign>=0,prior,0.),(1-PATH_CAP)*base)
    upper=np.minimum(np.where(sign<=0,prior,np.inf),(1+PATH_CAP)*base)
    bounds=list(zip(lower,upper))+[(0.,None)]*n
    scale=max(1e-12,float(np.max(np.abs(direction))))
    return linprog(np.r_[-direction/scale,np.zeros(n)],A_ub=inequality,
                   b_ub=np.r_[base,-base,BUDGET*base.sum()],
                   A_eq=equality,b_eq=M@base,bounds=bounds,method='highs')


def main():
    source=d.OUT/'task4_budgeted_source_direction/by_case.csv'
    out=d.OUT/'task4_budgeted_source_path_cap'
    d.save(out/'protocol.json',dict(
        scope='Posthoc control on 40 reused artificial cases. Does not select or validate an official candidate.',
        design='For the same clipped-source objective, cap each path movement at 5% of baseline flow in addition to the original global budget. Compare against its original global-only direction scaled to the same realized L1 movement.',
        path_cap=PATH_CAP,global_cap=BUDGET,source_sha256=d.sha(__file__),case_table_sha256=d.sha(source)))
    rows,failures=[],[]
    table=pd.read_csv(source).query("method=='clipped_source'")
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for row in group.itertuples():
            filename=hashlib.sha256(row.case.encode()).hexdigest()[:16]+'.npz'
            file=d.WORK/'task4_budgeted_source_direction_cases'/filename
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                base,prior,truth,count=z['baseline'],z['priors'][2],z['truth'],z['counts']
                result=capped_solution(base,prior,origin,dest,z['clipped_direction'])
                if not result.success:
                    failures.append(dict(case=row.case,seed=row.seed,message=result.message))
                    continue
                capped=result.x[:len(base)]
                movement=np.abs(capped-base).sum()
                assert np.max(np.abs(capped-base)-PATH_CAP*base)<=1e-7
                direction=z['clipped_source']-base
                factor=movement/max(1e-12,np.abs(direction).sum())
                assert factor<=1.+1e-9
                scaled=base+min(1.,factor)*direction
                np.testing.assert_allclose(np.abs(scaled-base).sum(),movement,rtol=1e-9,atol=1e-7)
                folder=d.WORK/'task4_budgeted_source_path_cap_cases'
                folder.mkdir(parents=True,exist_ok=True)
                artifact=folder/filename
                np.savez_compressed(artifact,capped=capped,scaled_global=scaled,seed=row.seed)
                digest=d.sha(artifact)
                reference=score(base,truth,prior,A,count,dest)
                for method,flow in [('path_cap',capped),('scaled_global',scaled)]:
                    assert flow.min()>=-1e-8
                    np.testing.assert_allclose(A@flow,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(flow-prior).sum(),np.abs(base-prior).sum(),rtol=1e-9)
                    od,total=score(flow,truth,prior,A,count,dest)
                    np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
                    rows.append(dict(panel=panel,case=row.case,noise=row.noise,seed=row.seed,method=method,
                                     relative_movement=float(np.abs(flow-base).sum()/base.sum()),
                                     moved_paths=int((np.abs(flow-base)>1e-9).sum()),od_delta=od-reference[0],
                                     artifact_sha256=digest))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_case.csv',index=False)
    frame.groupby(['noise','method']).agg(cases=('od_delta','size'),mean_od_delta=('od_delta','mean'),
        harms=('od_delta',lambda x:int((x< -1e-12).sum())),mean_moved_paths=('moved_paths','mean'),
        mean_movement=('relative_movement','mean')).to_csv(out/'summary.csv')
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_budgeted_source_path_cap_complete',
        next='Independently verify equal movement, the local cap and score invariants. Reused cases do not validate an official method.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
