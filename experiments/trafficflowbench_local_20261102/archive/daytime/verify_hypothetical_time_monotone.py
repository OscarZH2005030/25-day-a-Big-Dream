"""Verify bounded monotone projection optimality and complete portability."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from audit_hypothetical_time_monotone import enforce_origin_totals


def main():
    name='task4_hypothetical_time_monotone'
    out=d.OUT/name
    frame=pd.read_csv(out/'by_panel.csv')
    assert len(frame)==240
    max_violation=0.
    for panel,group in frame.groupby('panel'):
        paths=pd.read_csv(d.DATA/'corridors'/panel/'network/path_set.csv',dtype=str)
        origin,dest=[paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone','destination_zone']]
        origin-=origin.min()
        for spread,actions in group.groupby('spread'):
            file=d.WORK/'task4_hypothetical_time_conditioning_cases'/panel/f'spread{spread}.npz'
            with np.load(file) as z:
                truth=z['truth']
                totals=np.bincount(origin,weights=truth)
                for cutoff,subset in actions.groupby('cutoff'):
                    raw=z['reconstructed_od'][0 if cutoff<.01 else 1]
                    clipped,monotone,fallback=enforce_origin_totals(raw,origin,dest,totals)
                    for row in subset.itertuples():
                        flow={'raw':raw,'clip_and_scale':clipped,'isotonic':monotone}[row.method]
                        error=np.linalg.norm(flow-truth)/np.linalg.norm(truth)
                        np.testing.assert_allclose(error,row.relative_od_L2,rtol=1e-12)
                    assert subset[subset.method=='clip_and_scale'].uniform_fallback_origins.iloc[0]==fallback
                    for k,total in enumerate(totals):
                        ids=np.flatnonzero(origin==k)
                        ids=ids[np.argsort(dest[ids])]
                        if len(ids)<2:
                            continue
                        target=np.cumsum(raw[ids][::-1])[::-1][1:]
                        fitted=np.cumsum(monotone[ids][::-1])[::-1][1:]
                        gradient=fitted-target
                        # All vertices of the bounded decreasing chain are
                        # step functions with an initial block at the bound.
                        dot=total*np.r_[0.,np.cumsum(gradient)]-gradient@fitted
                        violation=max(0.,float(-dot.min()))/max(1.,total*np.linalg.norm(gradient,ord=1))
                        max_violation=max(max_violation,violation)
    assert max_violation<1e-9
    d.save(out/'independent_readback.json',dict(actions=240,max_normalized_projection_kkt_violation=max_violation,
        source_sha256=d.sha(__file__)))
    repeat=d.WORK/'publication_smoke/time_monotone'
    runner=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_hypothetical_time_monotone.py'
    d.state(status='running',stage='hypothetical_time_monotone_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['by_panel.csv','summary.csv']:
        assert (out/file).read_bytes()==(repeat/name/file).read_bytes()
    d.save(out/'portable_reproduction.json',dict(actions=240,exact_results=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='hypothetical_time_monotone_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
