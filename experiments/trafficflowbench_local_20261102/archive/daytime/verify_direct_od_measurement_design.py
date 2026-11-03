"""Reproduce the direct OD design by covariance downdates."""
import hashlib

import numpy as np
import pandas as pd

import common as d
from audit_direct_od_measurement_design import SEED, base, basis, indicators


def main():
    folder = d.OUT/'task4_direct_od_measurement_design'
    recorded = pd.read_csv(folder/'by_panel.csv',dtype={'seed':'uint64'})
    assert len(recorded)==400
    for (panel,split), group in recorded.groupby(['panel','split']):
        paths,A,_,prior,_ = base.load(panel,split)
        margins = [indicators(paths[k]).toarray() for k in ('origin_zone','destination_zone')]
        Q,rank = basis(np.vstack([A,*margins]).T)
        initial = np.eye(len(paths))-Q@Q.T
        weights = prior / np.sum(prior)
        seed = int.from_bytes(hashlib.sha256(f'{SEED}/{panel}/{split}'.encode()).digest()[:8],'little')
        assert (group.seed==seed).all()
        rng=np.random.default_rng(seed)
        orders={('uniform',i):rng.permutation(len(paths)) for i in range(8)}
        orders['top_prior',0]=np.argsort(-weights,kind='stable')
        orders['greedy_weighted_trace',0]=None
        for key, cells in group.groupby(['method','repeat']):
            expected=cells.set_index('budget')
            covariance=initial.copy()
            used=np.zeros(len(paths),dtype=bool)
            rank_gain=0
            for step in range(1,17):
                variance=covariance.diagonal().copy()
                order=orders[key]
                if order is None:
                    gain=np.divide(np.einsum('i,ij,ij->j',weights,covariance,covariance),variance,
                                   out=np.zeros_like(variance),where=variance>1e-12)
                    gain[used]=-1
                    index=int(np.argmax(gain))
                else:
                    index=int(order[step-1])
                used[index]=True
                if variance[index]>1e-12:
                    column=covariance[:,index].copy()
                    covariance -= np.outer(column,column)/variance[index]
                    rank_gain+=1
                if step in expected.index:
                    row=expected.loc[step]
                    assert row.starting_nullity==len(paths)-rank and row.rank_gain==rank_gain
                    np.testing.assert_allclose([covariance.trace()/initial.trace(),
                                                (weights@covariance.diagonal())/(weights@initial.diagonal())],
                                               [row.remaining_trace_ratio,row.remaining_weighted_trace_ratio],
                                               rtol=0,atol=1e-9)
    print('Verified 400 rows through independent covariance downdates')


if __name__=='__main__':
    main()
