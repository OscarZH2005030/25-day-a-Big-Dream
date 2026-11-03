"""Test OD ambiguity within the minimum invisible-L1 solution set."""
import hashlib
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_span_shrink_stress import load

SEED = 2026110345


def main():
    out=d.OUT/'task4_minimum_repair_face'
    source=d.OUT/'task4_minimum_invisible_repair/by_panel.csv'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
        scope='Feasible ambiguity on the frozen-net L1 repair optimum. No hidden labels, deployment or alternative submission.',
        constraints='Both gross margins of the saved optimum, nonnegativity and its minimum invisible L1 objective fixed. Use opposite fixed random linear objectives for two witnesses per group.',
        qualification='Different pair allocations can retain the same diagnostic objective. Original-prior L1 and hidden metrics need not agree.',
        source_sha256=d.sha(__file__),optimum_table_sha256=d.sha(source)))
    table=pd.read_csv(source).query("target=='frozen'")
    rows,failures,hashes=[],[],{}
    for row in table.itertuples():
        paths,A,_,prior,inputs=load(row.panel,row.split)
        hashes.update(inputs)
        oi,di=[pd.factorize(paths[key],sort=True)[0] for key in ['origin_zone','destination_zone']]
        size=len(paths)
        indexes=np.arange(size)
        O=sparse.csr_matrix((np.ones(size),(oi,indexes)),shape=(oi.max()+1,size))
        D=sparse.csr_matrix((np.ones(size),(di,indexes)),shape=(di.max()+1,size))
        M=sparse.vstack([O,D],format='csr')
        I=sparse.eye(size,format='csr')
        inequality=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format='csr')
        equality=sparse.vstack([
            sparse.hstack([M,sparse.csr_matrix(M.shape)]),
            sparse.csr_matrix(np.r_[np.zeros(size),np.ones(size)][None,:]),
        ],format='csr')
        file=d.WORK/'task4_minimum_invisible_repair_cases'/row.panel/row.split/'frozen.npz'
        assert d.sha(file)==row.artifact_sha256
        with np.load(file) as z:
            signed,reference=z['signed'],z['flow']
        optimum=float(np.abs(reference-signed).sum())
        rhs=np.r_[M@reference,optimum]
        bounds=np.r_[signed,-signed]
        seed=int.from_bytes(hashlib.sha256(f'{SEED}/{row.panel}/{row.split}'.encode()).digest()[:8],'little')%(2**63)
        direction=np.random.default_rng(seed).standard_normal(size)
        flows=[]
        for sign in [-1,1]:
            result=linprog(np.r_[sign*direction,np.zeros(size)],A_ub=inequality,b_ub=bounds,
                           A_eq=equality,b_eq=rhs,bounds=(0.,None),method='highs')
            if not result.success:
                failures.append(dict(panel=row.panel,split=row.split,seed=seed,sign=sign,message=result.message))
                continue
            f=result.x[:size]
            assert f.min()>=-1e-8
            np.testing.assert_allclose(M@f,M@reference,rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(A@f,A@reference,rtol=1e-10,atol=1e-7)
            np.testing.assert_allclose(np.abs(f-signed).sum(),optimum,rtol=1e-9)
            flows.append(f)
        if len(flows)!=2:
            continue
        folder=d.WORK/'task4_minimum_repair_face_cases'/row.panel
        folder.mkdir(parents=True,exist_ok=True)
        file=folder/f'{row.split}.npz'
        np.savez_compressed(file,lower=flows[1],upper=flows[0],seed=seed,objective=direction)
        rows.append(dict(panel=row.panel,split=row.split,seed=seed,
                         relative_od_separation=float(np.abs(flows[0]-flows[1]).sum()/reference.sum()),
                         relative_prior_l1_difference=float(abs(np.abs(flows[0]-prior).sum()-np.abs(flows[1]-prior).sum())/prior.sum()),
                         artifact_sha256=d.sha(file)))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame[['relative_od_separation','relative_prior_l1_difference']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_minimum_repair_face_complete',
        next='Independently verify tied-optimum witnesses and portable reproduction. These are diagnostic alternatives only.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
