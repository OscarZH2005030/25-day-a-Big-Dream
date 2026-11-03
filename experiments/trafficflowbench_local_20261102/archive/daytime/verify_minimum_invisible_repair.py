"""Check primal and dual feasibility and reproduce all diagnostic LPs."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def main():
    name='task4_minimum_invisible_repair'
    out=d.OUT/name
    table=pd.read_csv(out/'by_panel.csv')
    assert len(table)==40
    max_gap,max_dual_error=0.,0.
    for row in table.itertuples():
        paths,A,counts,prior,_=load(row.panel,row.split)
        origin,dest=[paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone','destination_zone']]
        origin,dest=origin-origin.min(),dest-origin.min()
        file=d.WORK/'task4_minimum_invisible_repair_cases'/row.panel/row.split/f'{row.target}.npz'
        assert d.sha(file)==row.artifact_sha256
        with np.load(file) as z:
            signed,flow,y=[z[k] for k in ['signed','flow','equality_dual']]
            upper,lower=np.split(z['inequality_dual'],2)
            assert max(upper.max(),lower.max())<1e-9
            y=np.r_[y,0.]
            path_dual=y[origin]-y[dest]
            # The lower-bound reduced costs for the flow and epigraph
            # variables independently certify dual feasibility.
            reduced_flow=-upper+lower-path_dual
            reduced_epigraph=1.+upper+lower
            violation=max(0.,float(-min(reduced_flow.min(),reduced_epigraph.min())))
            max_dual_error=max(max_dual_error,violation)
            assert violation<1e-8
            assert flow.min()>=-1e-8
            np.testing.assert_allclose(A@flow,A@signed,rtol=1e-10,atol=1e-7)
            if row.target=='released':
                np.testing.assert_allclose(A@flow,counts,rtol=1e-10,atol=1e-7)
            optimum=float(np.abs(flow-signed).sum())
            dual=float(signed@(upper-lower+path_dual))
            gap=abs(optimum-dual)/max(1.,optimum)
            max_gap=max(max_gap,gap)
            assert gap<1e-10
            np.testing.assert_allclose(optimum/prior.sum(),row.minimum_relative_L1,rtol=1e-12)
    d.save(out/'independent_readback.json',dict(cases=40,max_dual_violation=max_dual_error,max_relative_gap=max_gap,
        source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/minimum_invisible_repair'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_minimum_invisible_repair.py'
    d.state(status='running',stage='task4_minimum_invisible_repair_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_panel.csv','summary.csv','input_sha256.json','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for row in table.itertuples():
        file=repeat/'work/task4_minimum_invisible_repair_cases'/row.panel/row.split/f'{row.target}.npz'
        assert d.sha(file)==row.artifact_sha256
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_minimum_invisible_repair_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
