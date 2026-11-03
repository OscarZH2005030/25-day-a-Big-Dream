"""Check the full link operator by integer graph potentials and portable rerun."""
from collections import defaultdict, deque
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd


def main():
    name = 'task4_full_path_incidence'
    out = d.OUT/name
    results = pd.read_csv(out/'by_panel.csv')
    checks = []
    for row in results.itertuples():
        net = d.DATA/'corridors'/row.panel/'network'
        paths = pd.read_csv(net/'path_set.csv',dtype=str)
        incidence = pd.read_csv(net/'path_link_incidence.csv',dtype=str)
        links = {link:i for i,link in enumerate(incidence.link_id.unique())}
        path_indices = {path:i for i,path in enumerate(paths.path_id)}
        edge_values = np.zeros((len(paths),len(links)),dtype=np.int32)
        for pair in incidence.itertuples():
            edge_values[path_indices[pair.path_id],links[pair.link_id]]=1
        graph = defaultdict(list)
        edges = []
        for i,path in enumerate(paths.itertuples()):
            u,v = ('origin',path.origin_zone),('destination',path.destination_zone)
            graph[u].append((v,i,-1))
            graph[v].append((u,i,1))
            edges.append((u,v))
        potentials = {}
        components = 0
        for node in graph:
            if node in potentials:
                continue
            components += 1
            potentials[node]=np.zeros(len(links),dtype=np.int32)
            queue=deque([node])
            while queue:
                current=queue.popleft()
                for neighbor,edge,sign in graph[current]:
                    if neighbor not in potentials:
                        potentials[neighbor]=potentials[current]+sign*edge_values[edge]
                        queue.append(neighbor)
        residual=np.array([potentials[u]-potentials[v] for u,v in edges])-edge_values
        max_error=int(np.abs(residual).max())
        assert max_error==0 and row.maximum_integer_error==0 and row.rank_beyond_margins==0
        observed=set()
        for split in ['train','validation','private']:
            file=d.DATA/'task4'/row.panel/split/'synthetic_link_counts.csv'
            observed.update(pd.read_csv(file,usecols=['link_id'],dtype=str).link_id)
        assert len(links)==row.all_incidence_rows
        assert len(set(links)-observed)==row.uncounted_rows
        checks.append(dict(panel=row.panel,links=len(links),components=components,max_error=max_error))
    assert len(checks)==10
    d.save(out/'independent_readback.json',dict(checks=checks,source_sha256=d.sha(__file__),
        certificate='Integer origin/destination graph potentials reproduce every link indicator on every path.'))
    repeat=d.WORK/'publication_smoke/full_path_incidence'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_full_path_incidence.py'
    d.state(status='running',stage='full_path_incidence_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_panel.csv','failures.json','input_sha256.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    d.save(out/'portable_reproduction.json',dict(groups=10,exact_result_match=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='full_path_incidence_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
