"""Compare two exact-count L1 repair objectives on existing artificial cases."""
import hashlib
import msvcrt

import common as d
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from task4_span_shrink_stress import load, metrics


def main():
    source=d.OUT/'task4_source_clip_stress/all_cases.csv'
    out=d.OUT/'task4_l1_repair_stress'
    d.save(out/'protocol.json',dict(
        scope='Posthoc comparison on 40 reused artificial cases, no selection or new evaluation. Both alternatives match the same generated counts exactly.',
        methods='Minimum L1 to original prior versus minimum L1 to signed minimum-L2 net repair. Use deterministic HiGHS defaults and retain solver-selected ties. Frozen-style soft fit only a descriptive reference.',
        count_target='Net target recovered solely from the observed cuts, before truth is used for scoring.',
        source_sha256=d.sha(__file__),case_table_sha256=d.sha(source)))
    table=pd.read_csv(source).query("method=='baseline'")
    rows,failures,hashes=[],[],{}
    for panel,group in table.groupby('panel'):
        paths,A,_,_,inputs=load(panel,'train')
        hashes.update(inputs)
        origin,dest=[paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        n=int(dest.max()+1)
        size=len(paths)
        cols=np.arange(size)
        B=sparse.coo_matrix((np.r_[np.ones(size),-np.ones(size)],
                            (np.r_[origin,dest],np.r_[cols,cols])),shape=(n,size)).tocsr()
        cut_rows=[]
        for cut in range(n-1):
            match=np.flatnonzero(np.all(A==((origin<=cut)&(cut<dest)),axis=1))
            assert len(match)>0
            cut_rows.append(match[0])
        I=sparse.eye(size,format='csr')
        inequality=sparse.vstack([sparse.hstack([I,-I]),sparse.hstack([-I,-I])],format='csr')
        equality=sparse.hstack([B[:-1],sparse.csr_matrix((n-1,size))],format='csr')
        cost=np.r_[np.zeros(size),np.ones(size)]
        for row in group.itertuples():
            key=hashlib.sha256(row.case.encode()).hexdigest()[:16]
            file=d.WORK/'task4_source_clip_cases'/key/'case.npz'
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                count,b=z['counts'][2],z['prior'][2]
                truth=z['truth'][2]
                reference=z['baseline']
                net=np.diff(np.r_[0.,count[cut_rows],0.])
                error=net-B@b
                signed=b+(error[origin]-error[dest])/n
                flows={'soft_reference':reference}
                for method,anchor in [('prior_l1',b),('invisible_l1',signed)]:
                    result=linprog(cost,A_ub=inequality,b_ub=np.r_[anchor,-anchor],
                                   A_eq=equality,b_eq=net[:-1],bounds=(0.,None),method='highs')
                    if not result.success:
                        failures.append(dict(case=row.case,method=method,message=result.message))
                        continue
                    f=result.x[:size]
                    np.testing.assert_allclose(A@f,count,rtol=1e-10,atol=1e-7)
                    assert f.min()>=-1e-9
                    flows[method]=f
                folder=d.WORK/'task4_l1_repair_stress_cases'
                folder.mkdir(parents=True,exist_ok=True)
                artifact=folder/f'{key}.npz'
                np.savez_compressed(artifact,**flows)
                digest=d.sha(artifact)
                for method,flow in flows.items():
                    scores=metrics(flow,truth,b,A,count,dest)
                    rows.append(dict(panel=panel,case=row.case,noise=row.noise,method=method,
                                     source_artifact_sha256=row.artifact_sha256,artifact_sha256=digest,**scores))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'all_cases.csv',index=False)
    frame.groupby(['noise','method'])[['synthetic_S_od','synthetic_S_ODME','synthetic_S_dev']].mean().to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.save(out/'failures.json',failures)
    d.state(status='stage_complete',stage='task4_l1_repair_stress_complete',
        next='Verify matched-count methods and score readback. Reused artificial cases do not select a deployed objective.')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
