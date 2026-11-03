"""Independently verify artificial score readback and the portable chain."""
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
    name='task4_l1_repair_stress'
    out=d.OUT/name
    frame=pd.read_csv(out/'all_cases.csv')
    assert len(frame)==120 and frame.case.nunique()==40
    hashes={}
    maximum=0.
    for panel,group in frame.groupby('panel'):
        paths,A,_,_,_=load(panel,'train')
        dest=pd.factorize(paths.destination_zone,sort=True)[0]
        for case,rows in group.groupby('case'):
            key=hashlib.sha256(case.encode()).hexdigest()[:16]
            original=d.WORK/'task4_source_clip_cases'/key/'case.npz'
            file=d.WORK/'task4_l1_repair_stress_cases'/f'{key}.npz'
            assert set(rows.source_artifact_sha256)=={d.sha(original)}
            hashes[file.name]=d.sha(file)
            assert set(rows.artifact_sha256)=={hashes[file.name]}
            with np.load(original) as z,np.load(file) as candidates:
                truth,prior,count=[z[field][2] for field in ['truth','prior','counts']]
                for row in rows.itertuples():
                    flow=candidates[row.method]
                    assert flow.min()>=-1e-9
                    if row.method!='soft_reference':
                        np.testing.assert_allclose(A@flow,count,rtol=1e-10,atol=1e-7)
                    od,total=score(flow,truth,prior,A,count,dest)
                    maximum=max(maximum,abs(od-row.synthetic_S_od),abs(total-row.synthetic_S_ODME))
    assert maximum<1e-12
    d.save(out/'independent_readback.json',dict(actions=120,max_score_error=maximum,source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/l1_repair_stress'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_l1_repair.py'
    d.state(status='running',stage='task4_l1_repair_stress_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['all_cases.csv','summary.csv','input_sha256.json','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for filename,digest in hashes.items():
        assert d.sha(repeat/'work/task4_l1_repair_stress_cases'/filename)==digest
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_results=True,
        entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_l1_repair_stress_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
