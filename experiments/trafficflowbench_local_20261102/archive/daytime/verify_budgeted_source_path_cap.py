"""Check equal budgets, local path caps and portable reproduction."""
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
    name='task4_budgeted_source_path_cap'
    out=d.OUT/name
    table=pd.read_csv(out/'by_case.csv')
    assert len(table)==80 and table.case.nunique()==40
    hashes={}
    errors=[]
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for case,actions in group.groupby('case'):
            file=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
            original=d.WORK/'task4_budgeted_source_direction_cases'/file
            artifact=d.WORK/'task4_budgeted_source_path_cap_cases'/file
            hashes[file]=d.sha(artifact)
            assert set(actions.artifact_sha256)=={hashes[file]}
            with np.load(original) as z,np.load(artifact) as result:
                base,b,truth,count=z['baseline'],z['priors'][2],z['truth'],z['counts']
                capped,scaled=result['capped'],result['scaled_global']
                assert np.max(np.abs(capped-base)-.05*base)<=1e-7
                np.testing.assert_allclose(np.abs(capped-base).sum(),np.abs(scaled-base).sum(),rtol=1e-9,atol=1e-7)
                reference=score(base,truth,b,A,count,dest)
                for row in actions.itertuples():
                    f=capped if row.method=='path_cap' else scaled
                    assert f.min()>=-1e-8 and np.abs(f-base).sum()<=.002*base.sum()+1e-7
                    for index in [origin,dest]:
                        np.testing.assert_allclose(np.bincount(index,weights=f),np.bincount(index,weights=base),rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-b).sum(),np.abs(base-b).sum(),rtol=1e-9)
                    od,total=score(f,truth,b,A,count,dest)
                    np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
                    errors.append(abs(od-reference[0]-row.od_delta))
    assert max(errors)<1e-12
    d.save(out/'independent_readback.json',dict(actions=80,max_score_error=max(errors),source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/budgeted_source_path_cap'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_budgeted_source_path_cap.py'
    d.state(status='running',stage='task4_budgeted_source_path_cap_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_case.csv','summary.csv','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for filename,digest in hashes.items():
        assert d.sha(repeat/'work/task4_budgeted_source_path_cap_cases'/filename)==digest
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_budgeted_source_path_cap_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
