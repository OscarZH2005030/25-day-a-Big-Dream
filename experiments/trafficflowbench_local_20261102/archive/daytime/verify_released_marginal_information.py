"""Verify released count row space using exact net-zone potentials."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def modular_rank(matrix, prime=65521):
    a=np.asarray(matrix,dtype=np.int64).copy()%prime
    rank=0
    for col in range(a.shape[1]):
        candidates=np.flatnonzero(a[rank:,col])
        if not len(candidates):
            continue
        pivot=rank+int(candidates[0])
        a[[rank,pivot]]=a[[pivot,rank]]
        a[rank]=(a[rank]*pow(int(a[rank,col]),-1,prime))%prime
        a[rank+1:]=(a[rank+1:]-a[rank+1:,col,None]*a[rank])%prime
        rank+=1
        if rank==len(a):
            break
    return rank


def main():
    name='task4_released_marginal_information'
    out=d.OUT/name
    table=pd.read_csv(out/'by_zone.csv')
    checks=[]
    for (panel,split),group in table.groupby(['panel','split']):
        paths,A,_,_,_=load(panel,split)
        A=A.astype(np.int64)
        origin,dest=[paths[key].str[1:].astype(int).to_numpy() for key in ['origin_zone','destination_zone']]
        zones=np.unique(np.r_[origin,dest])
        first,last=int(zones[0]),int(zones[-1])
        lookup=dict(zip(zip(origin,dest),range(len(paths))))
        potential={first:np.zeros(len(A),dtype=np.int64)}
        potential.update({int(zone):-A[:,lookup[first,zone]] for zone in zones[1:]})
        reconstruction=np.column_stack([potential[o]-potential[v] for o,v in zip(origin,dest)])
        np.testing.assert_array_equal(reconstruction,A)
        rank=modular_rank(np.column_stack(list(potential.values())))
        assert rank==len(zones)-1
        assert (group.count_rank==rank).all()
        for row in group.itertuples():
            zone=int(row.zone[1:]) if row.zone!='all' else None
            expected=row.query=='net' or (row.query=='origin' and zone==first) or (row.query=='destination' and zone==last)
            assert row.identified==expected
            if not expected:
                middle=zone if zone is not None else int(zones[1])
                i,j,k=[lookup[pair] for pair in [(first,middle),(middle,last),(first,last)]]
                assert np.array_equal(A[:,i]+A[:,j],A[:,k])
        checks.append(dict(panel=panel,split=split,queries=len(group),exact_net_rank=rank,
                           integer_reconstruction_error=0))
    assert len(checks)==30
    d.save(out/'independent_readback.json',dict(groups=30,checks=checks,prime=65521,source_sha256=d.sha(__file__),
        certificate='Exact difference of common zone potentials gives rank upper bound. Modular elimination certifies matching lower bound. Triangle exchanges exclude interior gross margins and total demand.'))
    repeat=d.WORK/'publication_smoke/released_marginal_information'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_released_marginal_information.py'
    d.state(status='running',stage='released_marginal_information_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['by_zone.csv','by_panel.csv','summary.csv','failures.json','input_sha256.json']:
        assert (out/file).read_bytes()==(repeat/name/file).read_bytes()
    d.save(out/'portable_reproduction.json',dict(groups=30,exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='released_marginal_information_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
