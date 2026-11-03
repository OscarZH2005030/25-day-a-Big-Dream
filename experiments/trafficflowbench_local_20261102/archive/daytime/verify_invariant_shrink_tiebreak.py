"""Independently check the tied optimum and reproduce the secondary LP."""
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
    name='task4_invariant_shrink_tiebreak'
    out=d.OUT/name
    table=pd.read_csv(out/'by_case.csv')
    assert len(table)==160 and table.case.nunique()==40
    hashes={}
    max_error=0.
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for case,actions in group.groupby('case'):
            file=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
            original=d.WORK/'task4_invariant_invisible_shrink_cases'/file
            artifact=d.WORK/'task4_invariant_shrink_tiebreak_cases'/file
            hashes[file]=d.sha(artifact)
            assert set(actions.artifact_sha256)=={hashes[file]}
            with np.load(original) as z,np.load(artifact) as result:
                base,prior,truth,count=[z[key] for key in ['baseline','prior','truth','counts']]
                default,closest,signed=z['proposed'],result['closest'],z['signed']
                np.testing.assert_allclose(np.abs(default-signed).sum(),np.abs(closest-signed).sum(),rtol=1e-9)
                assert np.abs(closest-base).sum()<=np.abs(default-base).sum()+1e-7
                reference=score(base,truth,prior,A,count,dest)
                for row in actions.itertuples():
                    end=default if row.method=='default_tie' else closest
                    f=(1-row.blend)*base+row.blend*end
                    assert f.min()>=-1e-8
                    for index in [origin,dest]:
                        np.testing.assert_allclose(np.bincount(index,weights=f),np.bincount(index,weights=base),rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-prior).sum(),np.abs(base-prior).sum(),rtol=1e-9)
                    od,total=score(f,truth,prior,A,count,dest)
                    np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
                    max_error=max(max_error,abs(od-reference[0]-row.od_delta))
    assert max_error<1e-12
    d.save(out/'independent_readback.json',dict(actions=160,max_score_error=max_error,source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/invariant_shrink_tiebreak'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_invariant_shrink_tiebreak.py'
    d.state(status='running',stage='task4_invariant_shrink_tiebreak_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['by_case.csv','summary.csv','failures.json']:
        assert (out/file).read_bytes()==(repeat/name/file).read_bytes()
    for file,digest in hashes.items():
        assert d.sha(repeat/'work/task4_invariant_shrink_tiebreak_cases'/file)==digest
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_invariant_shrink_tiebreak_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
