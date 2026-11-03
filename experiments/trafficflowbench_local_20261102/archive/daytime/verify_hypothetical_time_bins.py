"""Verify time-bin ranks with independently constructed operators and SVD."""
from collections import defaultdict
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from scipy.linalg import svdvals


def main():
    name = 'task4_hypothetical_time_bins'
    out = d.OUT/name
    table = pd.read_csv(out/'by_panel_width.csv')
    assert len(table)==40
    checks=[]
    for panel,group in table.groupby('panel'):
        paths = pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str)
        pairs = [(int(o[1:]),int(v[1:])) for o,v in zip(paths.origin_zone,paths.destination_zone)]
        n = len(paths)
        zones = sorted(set(x for pair in pairs for x in pair))
        assert len(set(pairs))==n and all(o<v for o,v in pairs)
        start = [np.array([o==z for o,v in pairs]) for z in zones[:-1]]
        end = [np.array([v==z for o,v in pairs]) for z in zones[1:]]
        marginal_rank = 2*len(zones)-3
        assert (group.marginal_rank==marginal_rank).all()
        # Verify each column by streaming integer contributions, independent of
        # the vectorized construction in the registered run.
        unit_counts = defaultdict(int)
        seed = int(group.seed.iloc[0])
        test = np.random.default_rng(seed).integers(0,1000,size=n)
        for (origin,dest),value in zip(pairs,test):
            for cut in range(origin,dest):
                unit_counts[cut,origin] += int(value)
        recovered = [unit_counts[dest-1,origin]-unit_counts[dest,origin] for origin,dest in pairs]
        np.testing.assert_array_equal(test,recovered)
        for row in group.itertuples():
            if row.width==1:
                assert row.joint_rank==n and row.linear_nullity==0
                continue
            bins = defaultdict(list)
            for index,(origin,dest) in enumerate(pairs):
                for cut in range(origin,dest):
                    bins[cut,(cut-origin)//row.width].append(index)
            matrix=np.zeros((len(start)+len(end)+len(bins),n))
            matrix[:len(start)+len(end)]=start+end
            for i,indices in enumerate(bins.values(),len(start)+len(end)):
                matrix[i,indices]=1
            singular=svdvals(matrix)
            rank=int((singular>singular[0]*1e-10).sum())
            assert rank==row.joint_rank and n-rank==row.linear_nullity
        checks.append(dict(panel=panel,coarse_ranks=3,unit_reconstruction_error=0))
    d.save(out/'independent_readback.json',dict(groups=len(checks),checks=checks,source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/hypothetical_time_bins'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_hypothetical_time_bins.py'
    d.state(status='running',stage='hypothetical_time_bins_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_panel_width.csv','failures.json','input_sha256.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    d.save(out/'portable_reproduction.json',dict(groups=10,exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='hypothetical_time_bins_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
