"""Verify stored errors and independently recompute the regularized inverse."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from scipy.linalg import svd


def main():
    name='task4_hypothetical_time_conditioning'
    out=d.OUT/name
    frame=pd.read_csv(out/'by_panel.csv')
    hashes={}
    max_error=0.
    for panel,group in frame.groupby('panel'):
        paths=pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str)
        origin=np.array([int(o[1:]) for o in paths.origin_zone])
        dest=np.array([int(v[1:]) for v in paths.destination_zone])
        origin,dest=origin-origin.min(),dest-origin.min()
        n=int(dest.max()+1)
        times=np.arange(-24,n+24)
        for spread,actions in group.groupby('spread'):
            rel=f'{panel}/spread{spread}.npz'
            file=d.WORK/'task4_hypothetical_time_conditioning_cases'/rel
            hashes[rel]=d.sha(file)
            assert set(actions.artifact_sha256)=={hashes[rel]}
            with np.load(file) as z:
                truth,downstream,noise,od=[z[k] for k in ['truth','downstream','perturbation','reconstructed_od']]
                np.testing.assert_array_equal(downstream[dest-1,origin]-downstream[dest,origin],truth)
                np.testing.assert_allclose(np.linalg.norm(noise,axis=1),1.,rtol=1e-12)
                restored=np.zeros_like(downstream)
                for cut in range(n-1):
                    offsets=times[:,None]-cut+np.arange(cut+1)
                    if spread==0:
                        operator=np.equal(offsets,0).astype(float)
                    else:
                        operator=np.exp(-offsets**2/(2*spread**2))
                        operator/=operator.sum(axis=0)
                    clean=operator@downstream[cut,:cut+1]
                    perturbation=.01*np.linalg.norm(clean)*noise[cut]
                    np.testing.assert_allclose(np.linalg.norm(perturbation)/np.linalg.norm(clean),.01,rtol=1e-12)
                    u,s,v=svd(operator,full_matrices=False,lapack_driver='gesvd')
                    keep=s>=s[0]*.01
                    restored[cut,:cut+1]=v[keep].T@((u[:,keep].T@(clean+perturbation))/s[keep])
                    np.testing.assert_allclose(keep.mean(),z['retained_fractions'][1,cut],rtol=0,atol=1e-12)
                check=restored[dest-1,origin]-restored[dest,origin]
                error=float(np.max(np.abs(check-od[1]))/max(1.,np.max(np.abs(od[1]))))
                max_error=max(max_error,error)
                for row in actions.itertuples():
                    values=od[0 if row.cutoff<.01 else 1]
                    np.testing.assert_allclose(np.linalg.norm(values-truth)/np.linalg.norm(truth),row.relative_od_L2,rtol=1e-12)
                    assert abs((values<0).mean()-row.negative_od_fraction)<1e-12
    assert len(frame)==80 and max_error<1e-9
    d.save(out/'independent_readback.json',dict(actions=80,regularized_inverse_checks=40,max_relative_inverse_error=max_error,
        source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/time_conditioning'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_hypothetical_time_conditioning.py'
    d.state(status='running',stage='hypothetical_time_conditioning_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['by_panel.csv','summary.csv','input_sha256.json','failures.json']:
        assert (out/file).read_bytes()==(repeat/name/file).read_bytes()
    for rel,digest in hashes.items():
        assert d.sha(repeat/'work/task4_hypothetical_time_conditioning_cases'/rel)==digest
    d.save(out/'portable_reproduction.json',dict(matching_artifact_hashes=len(hashes),exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='hypothetical_time_conditioning_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
