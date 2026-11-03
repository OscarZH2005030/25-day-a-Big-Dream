"""Audit the fixed artificial shape constraints and portable rerun."""
import hashlib
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
import task4_span_shrink_stress as base
from verify_cycle_budget import score


def main():
    name = 'task4_public_range_shape_stress'
    out = d.OUT/name
    rows = pd.read_csv(out/'all_cases.csv')
    assert len(rows)==240 and rows.case.nunique()==80
    errors = []
    for panel, group in rows.groupby('panel'):
        paths,A,_,_,_ = base.load(panel,'train')
        dest = pd.factorize(paths.destination_zone,sort=True)[0]
        for case, actions in group.groupby('case'):
            key = hashlib.sha256(case.encode()).hexdigest()[:16]
            file = d.WORK/'task4_public_range_shape_cases'/key/'case.npz'
            assert set(actions.artifact_sha256)=={d.sha(file)}
            with np.load(file) as z:
                loaded = z['supplied_prior'] @ A.T
                observed = z['counts']
                np.testing.assert_allclose(loaded.sum(axis=1),observed.sum(axis=1),rtol=1e-12)
                tv = .5*np.abs(loaded/loaded.sum(axis=1,keepdims=True)-observed/observed.sum(axis=1,keepdims=True)).sum(axis=1)
                np.testing.assert_allclose(tv,.14,rtol=0,atol=1e-9)
                for row in actions.itertuples():
                    od,total = score(z[row.method],z['truth'][2],z['supplied_prior'][2],A,observed[2],dest)
                    errors.extend([abs(od-row.synthetic_S_od),abs(total-row.synthetic_S_ODME)])
    assert max(errors)<1e-12
    d.save(out/'independent_readback.json',dict(actions=240,max_score_error=max(errors),source_sha256=d.sha(__file__)))
    repeat = d.WORK/'publication_smoke/public_range_shape'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_public_range_shape.py'
    d.state(status='running',stage='public_range_shape_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['all_cases.csv','summary.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/file),pd.read_csv(repeat/name/file),check_exact=True)
    assert (out/'failures.json').read_bytes()==(repeat/name/'failures.json').read_bytes()
    for case, group in rows.groupby('case'):
        key = hashlib.sha256(case.encode()).hexdigest()[:16]
        assert set(group.artifact_sha256)=={d.sha(repeat/'work/task4_public_range_shape_cases'/key/'case.npz')}
    d.save(out/'portable_reproduction.json',dict(cases=80,matching_artifact_hashes=80,exact_csv_match=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='public_range_shape_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
