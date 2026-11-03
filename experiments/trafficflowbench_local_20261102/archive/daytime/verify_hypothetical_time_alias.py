"""Verify origin time aliasing via connected components of the actual edge set."""
import msvcrt
import subprocess
import sys
from collections import defaultdict

import common as d
import numpy as np
import pandas as pd


def graph_rank(pairs,offset):
    graph=defaultdict(set)
    for origin,dest in pairs:
        left=('o',origin)
        right=('d',offset[origin]-origin,dest)
        graph[left].add(right)
        graph[right].add(left)
    remaining=set(graph)
    components=0
    while remaining:
        components+=1
        stack=[remaining.pop()]
        while stack:
            node=stack.pop()
            neighbors=graph[node]&remaining
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    return len(graph)-components


def main():
    name='task4_hypothetical_time_alias'
    out=d.OUT/name
    table=pd.read_csv(out/'by_panel.csv')
    for panel,group in table.groupby('panel'):
        paths=pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str)
        pairs=[(int(o[1:]),int(v[1:])) for o,v in zip(paths.origin_zone,paths.destination_zone)]
        start=min(o for o,v in pairs)
        pairs=[(o-start,v-start) for o,v in pairs]
        n=max(v for o,v in pairs)
        rng=np.random.default_rng(int(group.seed.iloc[0]))
        rng.integers(0,1000,size=len(paths))
        offsets={'simultaneous':np.zeros(n,dtype=int),'jitter4':rng.integers(-4,5,n),
                 'half_compensation':np.arange(n)//2,'full_compensation':np.arange(n)}
        for row in group.itertuples():
            offset=offsets[row.scenario]
            rank=graph_rank(pairs,offset)
            assert rank==row.exact_rank and len(paths)-rank==row.linear_nullity
            assert len(set(offset-np.arange(n)))==row.streams
    assert len(table)==40
    d.save(out/'independent_readback.json',dict(rank_checks=40,method='Bipartite graph component traversal on path edge set.',source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/hypothetical_time_alias'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_hypothetical_time_alias.py'
    d.state(status='running',stage='hypothetical_time_alias_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['by_panel.csv','summary.csv','failures.json','input_sha256.json']:
        assert (out/file).read_bytes()==(repeat/name/file).read_bytes()
    d.save(out/'portable_reproduction.json',dict(cases=40,exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='hypothetical_time_alias_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
