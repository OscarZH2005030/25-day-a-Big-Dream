"""Read back the paired invariants and reproduce the frozen experiment."""
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
    name = 'task4_count_invariant_drift'
    out = d.OUT/name
    frame = pd.read_csv(out/'all_cases.csv')
    assert len(frame)==240 and frame.case.nunique()==40
    errors, invariants, hashes = [], [], {}
    for panel, group in frame.groupby('panel'):
        paths,A,_,_,_ = base.load(panel,'train')
        origin = pd.factorize(paths.origin_zone,sort=True)[0]
        dest = pd.factorize(paths.destination_zone,sort=True)[0]
        for case, actions in group.groupby('case'):
            key = hashlib.sha256(case.encode()).hexdigest()[:16]
            folder = d.WORK/'task4_count_invariant_drift_cases'/key
            with np.load(folder/'moderate.npz') as original, np.load(folder/'repaired.npz') as repaired:
                for field in ['seed','counts','fold','cycles']:
                    np.testing.assert_array_equal(original[field],repaired[field])
                np.testing.assert_array_equal(original['prior'][:2],repaired['prior'][:2])
                change = repaired['truth']-original['truth']
                error = max(np.max(np.abs(A@change)),np.max(np.abs(np.bincount(origin,weights=change))),
                            np.max(np.abs(np.bincount(dest,weights=change))))
                invariants.append(float(error/max(1.,original['truth'].sum())))
                for variant,z in [('moderate',original),('repaired',repaired)]:
                    rel = f'{key}/{variant}.npz'
                    hashes[rel] = d.sha(folder/f'{variant}.npz')
                    subset = actions[actions.variant==variant]
                    assert set(subset.artifact_sha256)=={hashes[rel]}
                    loaded = z['prior']@A.T
                    counts = z['counts']
                    np.testing.assert_allclose(loaded.sum(axis=1),counts.sum(axis=1),rtol=1e-12)
                    tv = .5*np.abs(loaded/loaded.sum(axis=1,keepdims=True)-counts/counts.sum(axis=1,keepdims=True)).sum(axis=1)
                    np.testing.assert_allclose(tv,.14,rtol=0,atol=1e-9)
                    for row in subset.itertuples():
                        od,total = score(z[row.method],z['truth'],z['prior'][2],A,counts[2],dest)
                        errors.extend([abs(od-row.synthetic_S_od),abs(total-row.synthetic_S_ODME)])
    assert max(errors)<1e-12 and max(invariants)<1e-10
    d.save(out/'independent_readback.json',dict(actions=240,paired_cases=40,
        max_score_error=max(errors),max_relative_invariance_error=max(invariants),source_sha256=d.sha(__file__)))
    repeat = d.WORK/'publication_smoke/count_invariant_drift'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_count_invariant_drift.py'
    d.state(status='running',stage='count_invariant_drift_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['all_cases.csv','summary.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/filename),pd.read_csv(repeat/name/filename),check_exact=True)
    assert (out/'failures.json').read_bytes()==(repeat/name/'failures.json').read_bytes()
    for rel,digest in hashes.items():
        assert d.sha(repeat/'work/task4_count_invariant_drift_cases'/rel)==digest
    d.save(out/'portable_reproduction.json',dict(paired_cases=40,exact_csv_match=True,
        matching_artifact_hashes=len(hashes),entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='count_invariant_drift_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
