"""Verify the net-space projection independently against a count-space SVD."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from scipy.linalg import svd
from task4_span_shrink_stress import load


def main():
    name='task4_net_repair'
    out=d.OUT/name
    rows=pd.read_csv(out/'by_panel.csv')
    anchor=d.ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv'
    frozen=pd.read_csv(anchor,dtype={'panel':str,'path_id':str,'departure_time':str})
    errors=[]
    for row in rows.itertuples():
        paths,A,c,b,_=load(row.panel,row.split)
        file=d.WORK/'task4_net_repair_cases'/row.panel/f'{row.split}.npz'
        assert d.sha(file)==row.artifact_sha256
        priorfile=d.DATA/'task4'/row.panel/row.split/'synthetic_weak_prior.csv'
        departure=pd.read_csv(priorfile,usecols=['departure_time'],dtype=str).departure_time.iloc[0]
        flow=frozen[(frozen.panel==row.panel)&(frozen.departure_time==departure)].set_index('path_id').loc[paths.path_id,'path_flow'].to_numpy()
        u,s,vt=svd(A,full_matrices=False)
        keep=s>s[0]*1e-10
        target=c-A@b
        solution=vt[keep].T@((u[:,keep].T@target)/s[keep])
        difference=flow-b
        visible=vt[keep].T@(vt[keep]@difference)
        invisible=difference-visible
        with np.load(file) as z:
            for k,expected in [('minnorm',solution),('visible',visible),('invisible',invisible)]:
                errors.append(float(np.max(np.abs(z[k]-expected))/max(1.,np.max(np.abs(expected)))))
            assert abs(np.mean(b+solution<0)-row.signed_repair_negative_fraction)<1e-12
            np.testing.assert_allclose((invisible@invisible)/(difference@difference),
                                       row.frozen_change_invisible_energy_fraction,rtol=1e-12)
    assert len(rows)==20 and max(errors)<1e-10
    d.save(out/'independent_readback.json',dict(groups=20,max_relative_projection_error=max(errors),source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/net_repair'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_task4_net_repair.py'
    d.state(status='running',stage='task4_net_repair_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['by_panel.csv','input_sha256.json','failures.json']:
        assert (out/filename).read_bytes()==(repeat/name/filename).read_bytes()
    for row in rows.itertuples():
        file=repeat/'work/task4_net_repair_cases'/row.panel/f'{row.split}.npz'
        assert d.sha(file)==row.artifact_sha256
    d.save(out/'portable_reproduction.json',dict(groups=20,matching_artifact_hashes=20,exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='task4_net_repair_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
