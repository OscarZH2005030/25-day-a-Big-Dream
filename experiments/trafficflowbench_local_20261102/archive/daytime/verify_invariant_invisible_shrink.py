"""Verify the fixed non-OD components and reproduce the frozen stress."""
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
    name='task4_invariant_invisible_shrink'
    out=d.OUT/name
    table=pd.read_csv(out/'all_cases.csv')
    assert len(table)==120 and table.case.nunique()==40
    hashes={}
    max_invariant,max_score=0.,0.
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for case,rows in group.groupby('case'):
            filename=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
            file=d.WORK/'task4_invariant_invisible_shrink_cases'/filename
            hashes[filename]=d.sha(file)
            assert set(rows.artifact_sha256)=={hashes[filename]}
            with np.load(file) as z:
                base,proposal,prior,truth,count=[z[k] for k in ['baseline','proposed','prior','truth','counts']]
                reference=score(base,truth,prior,A,count,dest)
                assert np.abs(proposal-z['signed']).sum()<=np.abs(base-z['signed']).sum()+1e-7
                for row in rows.itertuples():
                    flow=(1-row.blend)*base+row.blend*proposal
                    assert flow.min()>=-1e-8
                    change=flow-base
                    invariant=max(np.max(np.abs(A@change)),np.max(np.abs(np.bincount(origin,weights=change))),
                                  np.max(np.abs(np.bincount(dest,weights=change))),
                                  abs(np.abs(flow-prior).sum()-np.abs(base-prior).sum()))
                    max_invariant=max(max_invariant,float(invariant))
                    od,total=score(flow,truth,prior,A,count,dest)
                    max_score=max(max_score,abs(od-row.synthetic_S_od),abs(total-row.synthetic_S_ODME))
                    np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
    assert max_invariant<1e-7 and max_score<1e-12
    d.save(out/'independent_readback.json',dict(actions=120,max_constraint_error=max_invariant,max_score_error=max_score,
        source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/invariant_invisible_shrink'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_invariant_invisible_shrink.py'
    d.state(status='running',stage='task4_invariant_invisible_shrink_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['all_cases.csv','summary.csv','input_sha256.json','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for filename,digest in hashes.items():
        assert d.sha(repeat/'work/task4_invariant_invisible_shrink_cases'/filename)==digest
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_invariant_invisible_shrink_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
