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
    name='task4_source_agreement_stress'
    out=d.OUT/name
    table=pd.read_csv(out/'by_case.csv')
    assert len(table)==240 and table.case.nunique()==80
    hashes={}
    errors=[]
    for panel,group in table.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        origin,dest=[pd.factorize(paths[k],sort=True)[0] for k in ['origin_zone','destination_zone']]
        for case,actions in group.groupby('case'):
            file=hashlib.sha256(case.encode()).hexdigest()[:16]+'.npz'
            artifact=d.WORK/'task4_source_agreement_stress_cases'/file
            hashes[file]=d.sha(artifact)
            assert set(actions.artifact_sha256)=={hashes[file]}
            with np.load(artifact) as z:
                base,b,truth,count=z['baseline'],z['priors'][2],z['truth'],z['counts']
                capped,scaled=z['agreement_source'],z['mean_source']
                for candidate in [capped,scaled]:
                    assert np.max(np.abs(candidate-base)-.05*base)<=1e-7
                np.testing.assert_allclose(np.abs(capped-base).sum(),np.abs(scaled-base).sum(),rtol=1e-9,atol=1e-7)
                reference=score(base,truth,b,A,count,dest)
                for row in actions.itertuples():
                    f=z[row.method]
                    assert f.min()>=-1e-8 and np.abs(f-base).sum()<=.002*base.sum()+1e-7
                    for index in [origin,dest]:
                        np.testing.assert_allclose(np.bincount(index,weights=f),np.bincount(index,weights=base),rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(A@f,A@base,rtol=1e-10,atol=1e-7)
                    np.testing.assert_allclose(np.abs(f-b).sum(),np.abs(base-b).sum(),rtol=1e-9)
                    od,total=score(f,truth,b,A,count,dest)
                    np.testing.assert_allclose(total-reference[1],.45*(od-reference[0]),rtol=0,atol=1e-10)
                    errors.append(abs(od-reference[0]-row.od_delta))
    assert max(errors)<1e-12
    d.save(out/'independent_readback.json',dict(actions=240,max_score_error=max(errors),source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/source_agreement'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_source_agreement.py'
    d.state(status='running',stage='task4_source_agreement_stress_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_case.csv','summary.csv','failures.json','input_sha256.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for filename,digest in hashes.items():
        assert d.sha(repeat/'work/task4_source_agreement_stress_cases'/filename)==digest
    d.save(out/'portable_reproduction.json',dict(cases=80,matching_artifact_hashes=80,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_source_agreement_stress_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
