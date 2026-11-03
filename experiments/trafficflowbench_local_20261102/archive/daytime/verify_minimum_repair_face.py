"""Verify tied-flow witnesses and reproduce the full portable chain."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_span_shrink_stress import load


def main():
    name='task4_minimum_repair_face'
    out=d.OUT/name
    rows=pd.read_csv(out/'by_panel.csv')
    assert len(rows)==20
    max_error=0.
    for row in rows.itertuples():
        paths,A,_,prior,_=load(row.panel,row.split)
        oi,di=[pd.factorize(paths[key],sort=True)[0] for key in ['origin_zone','destination_zone']]
        file=d.WORK/'task4_minimum_repair_face_cases'/row.panel/f'{row.split}.npz'
        assert d.sha(file)==row.artifact_sha256
        reference=d.WORK/'task4_minimum_invisible_repair_cases'/row.panel/row.split/'frozen.npz'
        with np.load(file) as z,np.load(reference) as anchor:
            low,high,direction=z['lower'],z['upper'],z['objective']
            ref,signed=anchor['flow'],anchor['signed']
            assert np.isfinite(low).all() and np.isfinite(high).all()
            assert min(low.min(),high.min())>=-1e-8
            assert direction@low<=direction@ref+1e-7 and direction@ref<=direction@high+1e-7
            for flow in [low,high]:
                for indices in [oi,di]:
                    a,b=np.bincount(indices,weights=flow),np.bincount(indices,weights=ref)
                    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-7)
                np.testing.assert_allclose(A@flow,A@ref,rtol=1e-10,atol=1e-7)
                objective=np.abs(ref-signed).sum()
                error=abs(np.abs(flow-signed).sum()-objective)/max(1.,objective)
                max_error=max(max_error,float(error))
                assert error<1e-9
            np.testing.assert_allclose(np.abs(low-high).sum()/ref.sum(),row.relative_od_separation,rtol=1e-12)
            prior_difference=abs(np.abs(low-prior).sum()-np.abs(high-prior).sum())/prior.sum()
            np.testing.assert_allclose(prior_difference,row.relative_prior_l1_difference,rtol=1e-10,atol=1e-13)
    d.save(out/'independent_readback.json',dict(groups=20,flows=40,max_relative_objective_error=max_error,
        source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/minimum_repair_face'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_minimum_repair_face.py'
    d.state(status='running',stage='task4_minimum_repair_face_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_panel.csv','summary.csv','input_sha256.json','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for row in rows.itertuples():
        file=repeat/'work/task4_minimum_repair_face_cases'/row.panel/f'{row.split}.npz'
        assert d.sha(file)==row.artifact_sha256
    d.save(out/'portable_reproduction.json',dict(groups=20,matching_artifact_hashes=20,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_minimum_repair_face_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
