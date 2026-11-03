"""Independently check matched budgets and score invariants, then reproduce."""
import hashlib
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load
from verify_cycle_budget import score


def main():
    name='task4_budgeted_source_direction'
    out=d.OUT/name
    table=pd.read_csv(out/'by_case.csv')
    assert len(table)==120 and table.case.nunique()==40
    hashes={}
    max_error,max_budget=0.,0.
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for case,actions in group.groupby('case'):
            filename=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
            file=d.WORK/'task4_budgeted_source_direction_cases'/filename
            hashes[filename]=d.sha(file)
            assert set(actions.artifact_sha256)=={hashes[filename]}
            with np.load(file) as z:
                base,prior,truth,count=z['baseline'],z['priors'][2],z['truth'],z['counts']
                ref=score(base,truth,prior,A,count,dest)
                for row in actions.itertuples():
                    f=z[row.method]
                    assert f.min()>=-1e-8
                    for idx in [origin,dest]:
                        np.testing.assert_allclose(np.bincount(idx,weights=f),np.bincount(idx,weights=base),rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-prior).sum(),np.abs(base-prior).sum(),rtol=1e-9)
                    movement=np.linalg.norm(f-base,ord=1)/base.sum()
                    expected=0. if row.method=='baseline' else .002
                    max_budget=max(max_budget,abs(movement-expected))
                    np.testing.assert_allclose(movement,row.relative_movement,rtol=1e-12,atol=1e-12)
                    od,total=score(f,truth,prior,A,count,dest)
                    np.testing.assert_allclose(total-ref[1],.45*(od-ref[0]),rtol=0,atol=1e-10)
                    max_error=max(max_error,abs(od-ref[0]-row.od_delta))
    assert max_error<1e-12 and max_budget<1e-10
    d.save(out/'independent_readback.json',dict(actions=120,max_score_error=max_error,max_budget_error=max_budget,
        source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/budgeted_source_direction'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_budgeted_source_direction.py'
    d.state(status='running',stage='task4_budgeted_source_direction_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_case.csv','summary.csv','input_sha256.json','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for filename,digest in hashes.items():
        assert d.sha(repeat/'work/task4_budgeted_source_direction_cases'/filename)==digest
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_budgeted_source_direction_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
