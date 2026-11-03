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
from task4_sparse_cutoff_stress import g


def main():
    name = 'task4_sparse_cutoff_stress'
    out = d.OUT/name
    rows = pd.read_csv(out/'all_cases.csv')
    assert len(rows)==80 and rows.case.nunique()==20
    errors, descent_errors, relative_descent_errors = [], [], []
    for panel, group in rows.groupby('panel'):
        paths,A,_,_,_ = base.load(panel,'train')
        X = g.family.structure.features(paths)['span_origin_destination']
        dest = pd.factorize(paths.destination_zone,sort=True)[0]
        for case, actions in group.groupby('case'):
            key = hashlib.sha256(case.encode()).hexdigest()[:16]
            file = d.WORK/'task4_sparse_cutoff_cases'/key/'case.npz'
            assert set(actions.artifact_sha256)=={d.sha(file)}
            with np.load(file) as z:
                loaded = z['prior'] @ A.T
                observed = z['counts']
                np.testing.assert_allclose(loaded.sum(axis=1),observed.sum(axis=1),rtol=1e-12)
                tv = .5*np.abs(loaded/loaded.sum(axis=1,keepdims=True)-observed/observed.sum(axis=1,keepdims=True)).sum(axis=1)
                np.testing.assert_allclose(tv,.14,rtol=0,atol=1e-9)
                prior = z['prior']
                source = np.mean([b/max(float(np.median(b[b>0])),1e-9) for b in prior[:2]],axis=0)
                design = np.column_stack([X,source-X@g.qr.ridge_qr(X,source)])
                for index, (fold, physical_cutoff, stored_initial, stored_final) in enumerate(z['cutoff_diagnostics']):
                    k = int(fold)
                    train = z['fold'] != k
                    target = prior[2,train]/z['scales'][k]
                    method = ['median_cutoff005','mean_floor005'][index%2]
                    cutoff = physical_cutoff/z['scales'][k]
                    expected = 1. if index%2==0 else max(1.,float(target.mean()))
                    np.testing.assert_allclose(cutoff,expected,rtol=1e-12)
                    def objective(coef):
                        residual = np.abs(design[train]@coef-target)
                        capped = np.minimum(residual,cutoff)
                        return .5*(capped@capped + coef[1:]@coef[1:]) + cutoff*(residual-capped).sum()
                    initial = objective(z['squared005_coef'][k])
                    final = objective(z[method+'_coef'][k])
                    assert final <= initial + 1e-9*max(1.,initial)
                    error = max(abs(initial-stored_initial),abs(final-stored_final))
                    descent_errors.append(error)
                    relative_descent_errors.append(error/max(1.,initial))
                for row in actions.itertuples():
                    od,total = score(z[row.method],z['truth'][2],z['prior'][2],A,observed[2],dest)
                    errors.extend([abs(od-row.synthetic_S_od),abs(total-row.synthetic_S_ODME)])
    d.save(out/'independent_readback.json',dict(actions=80,fold_objective_checks=200,
        max_objective_error=max(descent_errors),
        max_objective_error_relative_to_initial=max(relative_descent_errors),
        max_score_error=max(errors),source_sha256=d.sha(__file__)))
    assert max(errors)<1e-12 and max(relative_descent_errors)<1e-10
    repeat = d.WORK/'publication_smoke/sparse_cutoff'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_sparse_cutoff.py'
    d.state(status='running',stage='sparse_cutoff_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['all_cases.csv','summary.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/file),pd.read_csv(repeat/name/file),check_exact=True)
    assert (out/'failures.json').read_bytes()==(repeat/name/'failures.json').read_bytes()
    for case, group in rows.groupby('case'):
        key = hashlib.sha256(case.encode()).hexdigest()[:16]
        assert set(group.artifact_sha256)=={d.sha(repeat/'work/task4_sparse_cutoff_cases'/key/'case.npz')}
    d.save(out/'portable_reproduction.json',dict(cases=20,matching_artifact_hashes=20,exact_csv_match=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='sparse_cutoff_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
