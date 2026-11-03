"""Independently verify the rank table, witness invariants and portable run."""
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from scipy.linalg import svdvals
from task4_span_shrink_stress import load


def main():
    name = 'task4_extra_measurement_audit'
    out = d.OUT/name
    witnesses = pd.read_csv(out/'witness_summary.csv')
    ranks = pd.read_csv(out/'ranks.csv')
    assert len(witnesses)==40 and len(ranks)==80
    maximum = 0.
    for (panel,split),group in witnesses.groupby(['panel','split']):
        paths,A,_,prior,_ = load(panel,split)
        origin,dest = [paths[key].str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
                       for key in ['origin_zone','destination_zone']]
        span = dest-origin
        origins = np.unique(origin)[:,None]==origin
        destinations = np.unique(dest)[:,None]==dest
        histogram = np.unique(span)[:,None]==span
        existing = np.vstack([A,origins,destinations])
        matrices = {
            'counts_margins': existing,
            'plus_total_span': np.vstack([existing,span]),
            'plus_histogram': np.vstack([existing,histogram]),
            'plus_histogram_and_moments': np.vstack([existing,histogram,origins*span,destinations*span]),
        }
        np.testing.assert_array_equal(np.unique(dest)@destinations-np.unique(origin)@origins,span)
        expected = ranks[(ranks.panel==panel)&(ranks.split==split)].set_index('method')
        for method,matrix in matrices.items():
            singular = svdvals(matrix)
            rank = int((singular>singular[0]*1e-10).sum())
            assert rank==expected.loc[method,'rank']
        for row in group.itertuples():
            file = d.WORK/'task4_extra_measurement_witnesses'/panel/split/f'{row.method}.npz'
            assert d.sha(file)==row.artifact_sha256
            with np.load(file) as z:
                f,g = z['baseline'],z['witness']
                delta = g-f
                assert g.min()>=-1e-9
                matrix = matrices[row.method]
                error = np.max(np.abs(matrix@delta))/max(1.,np.max(np.abs(matrix@f)))
                maximum = max(maximum,float(error))
                assert error<1e-9
                np.testing.assert_allclose(np.abs(g-prior).sum(),np.abs(f-prior).sum(),rtol=1e-9)
                np.testing.assert_allclose(np.abs(delta).sum()/f.sum(),row.relative_od_movement,rtol=1e-12)
    d.save(out/'independent_readback.json',dict(rank_checks=80,witnesses=40,max_constraint_error=maximum,
        span_identity='Total span is destination weighted total minus origin weighted total.',
        source_sha256=d.sha(__file__)))
    repeat = d.WORK/'publication_smoke/extra_measurements'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_extra_measurements.py'
    d.state(status='running',stage='extra_measurement_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for filename in ['ranks.csv','summary.csv','witness_summary.csv']:
        a,b = [pd.read_csv(p/filename).drop(columns='old_witness_sha256',errors='ignore') for p in [out,repeat/name]]
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert (out/'failures.json').read_bytes()==(repeat/name/'failures.json').read_bytes()
    for row in witnesses.itertuples():
        file = repeat/'work/task4_extra_measurement_witnesses'/row.panel/row.split/f'{row.method}.npz'
        assert d.sha(file)==row.artifact_sha256
    d.save(out/'portable_reproduction.json',dict(matching_witness_hashes=40,exact_result_tables=True,
        scope='Portable anchors staged from the frozen prediction CSV, so anchor provenance hash differs.',
        baseline_sha256=d.sha(runner.with_name('frozen_reference_path_flow.csv')),entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='extra_measurement_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
