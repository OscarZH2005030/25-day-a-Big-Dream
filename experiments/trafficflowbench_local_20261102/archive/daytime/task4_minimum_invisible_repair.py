"""Bound the invisible correction required by nonnegativity at fixed net flow."""
import msvcrt
from datetime import datetime, timezone

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_span_shrink_stress import load


def main():
    source=d.OUT/'task4_net_repair/by_panel.csv'
    out=d.OUT/'task4_minimum_invisible_repair'
    anchor=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    d.save(out/'protocol.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),seed=None,
        scope='Diagnostic optimization on published inputs and the frozen prediction. No hidden scores or deployment.',
        objective='Minimize L1 distance to the signed minimum-L2 correction at fixed net vector, with nonnegative path flows. Solve for released and frozen net targets separately.',
        comparison='Only compare the frozen invisible L1 size to its own matched-net optimum. Released-count compatibility with the hidden reference is unverified.',
        source_sha256=d.sha(__file__),projection_table_sha256=d.sha(source),frozen_sha256=d.sha(anchor)))
    table=pd.read_csv(source)
    frozen=pd.read_csv(anchor,dtype={'panel':str,'path_id':str,'departure_time':str})
    rows,failures,hashes=[],[],{}
    for row in table.itertuples():
        paths,A,count,prior,inputs=load(row.panel,row.split)
        hashes.update(inputs)
        origin,dest=[paths[key].str[1:].astype(int).to_numpy() for key in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        n=int(dest.max()+1)
        size=len(paths)
        cols=np.arange(size)
        B=sparse.coo_matrix((np.r_[np.ones(size),-np.ones(size)],
                            (np.r_[origin,dest],np.r_[cols,cols])),shape=(n,size)).tocsr()[:-1]
        I=sparse.eye(size,format='csr')
        inequality=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format='csr')
        equality=sparse.hstack([B,sparse.csr_matrix(B.shape)],format='csr')
        objective=np.r_[np.zeros(size),np.ones(size)]
        path=d.WORK/'task4_net_repair_cases'/row.panel/f'{row.split}.npz'
        assert d.sha(path)==row.artifact_sha256
        priorfile=d.DATA/'task4'/row.panel/row.split/'synthetic_weak_prior.csv'
        departure=pd.read_csv(priorfile,usecols=['departure_time'],dtype=str).departure_time.iloc[0]
        frozen_flow=frozen[(frozen.panel==row.panel)&(frozen.departure_time==departure)].set_index('path_id').loc[paths.path_id,'path_flow'].to_numpy()
        with np.load(path) as z:
            targets={'released':prior+z['minnorm'],'frozen':prior+z['visible']}
        for target,signed in targets.items():
            rhs=B@signed
            bounds=np.r_[signed,-signed]
            result=linprog(objective,A_ub=inequality,b_ub=bounds,A_eq=equality,b_eq=rhs,
                           bounds=(0.,None),method='highs')
            if not result.success:
                failures.append(dict(panel=row.panel,split=row.split,target=target,message=result.message))
                continue
            flow=result.x[:size]
            change=flow-signed
            minimum=float(np.abs(change).sum())
            dual=float(bounds@result.ineqlin.marginals+rhs@result.eqlin.marginals)
            frozen_size=float(np.abs(frozen_flow-signed).sum()) if target=='frozen' else None
            assert flow.min()>=-1e-8
            assert np.max(np.abs(B@change))<1e-7
            assert abs(minimum-dual)<1e-7*max(1.,minimum)
            assert abs(minimum-result.fun)<1e-7*max(1.,minimum)
            if target=='released':
                np.testing.assert_allclose(A@flow,count,rtol=1e-10,atol=1e-7)
            else:
                assert minimum<=frozen_size+1e-7*max(1.,frozen_size)
                np.testing.assert_allclose(A@flow,A@frozen_flow,rtol=1e-10,atol=1e-7)
            folder=d.WORK/'task4_minimum_invisible_repair_cases'/row.panel/row.split
            folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'{target}.npz'
            np.savez_compressed(file,signed=signed,flow=flow,inequality_dual=result.ineqlin.marginals,
                                equality_dual=result.eqlin.marginals)
            rows.append(dict(panel=row.panel,split=row.split,target=target,
                             minimum_relative_L1=minimum/prior.sum(),
                             negative_mass_lower_bound=float(np.maximum(-signed,0.).sum()),
                             frozen_relative_L1=frozen_size/prior.sum() if frozen_size is not None else None,
                             optimum_to_frozen_ratio=minimum/frozen_size if frozen_size else None,
                             relative_dual_gap=abs(minimum-dual)/max(1.,minimum),artifact_sha256=d.sha(file)))
    assert len(rows)+len(failures)==40
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_panel.csv',index=False)
    frame.groupby('target')[['minimum_relative_L1','frozen_relative_L1','optimum_to_frozen_ratio']].agg(['min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_minimum_invisible_repair_complete',
        next='Verify the LP certificate and portable reproduction. Lower invisible L1 is not evidence of more accurate hidden OD.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
