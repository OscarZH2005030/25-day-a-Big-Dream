"""Verify matched movement, invariants and portable reproduction."""
import hashlib
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def main():
    name='task4_equal_movement_ties'
    out=d.OUT/name
    table=pd.read_csv(out/'by_case.csv')
    assert len(table)==160 and table.case.nunique()==40
    errors=[]
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        for case,actions in group.groupby('case'):
            file=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
            with (
                np.load(d.WORK/'task4_invariant_invisible_shrink_cases'/file) as z,
                np.load(d.WORK/'task4_invariant_shrink_tiebreak_cases'/file) as secondary,
            ):
                base,prior,truth=z['baseline'],z['prior'],z['truth']
                default=z['proposed']-base
                closest=secondary['closest']-base
                default_size=np.linalg.norm(default,ord=1)
                closest_size=np.linalg.norm(closest,ord=1)
                scale=closest_size/default_size if default_size>1e-12 else 0.
                assert scale<=1.+1e-9
                baseline=max(0.,1.-np.linalg.norm(base-truth,ord=1)/truth.sum())
                for row in actions.itertuples():
                    step=closest if row.method=='minimum_move' else scale*default
                    prediction=base+row.fraction*step
                    assert prediction.min()>=-1e-8
                    np.testing.assert_allclose(A@prediction,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.linalg.norm(prediction-prior,ord=1),np.linalg.norm(base-prior,ord=1),rtol=1e-9)
                    np.testing.assert_allclose(row.relative_movement,row.fraction*closest_size/base.sum(),rtol=1e-9,atol=1e-12)
                    od=max(0.,1.-np.linalg.norm(prediction-truth,ord=1)/truth.sum())
                    errors.append(abs(od-baseline-row.od_delta))
    assert max(errors)<1e-12
    d.save(out/'independent_readback.json',dict(actions=160,max_score_error=max(errors),source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/equal_movement_ties'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_equal_movement_ties.py'
    d.state(status='running',stage='task4_equal_movement_ties_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_case.csv','paired_od.csv','summary.csv']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    d.save(out/'portable_reproduction.json',dict(actions=160,exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_equal_movement_ties_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
