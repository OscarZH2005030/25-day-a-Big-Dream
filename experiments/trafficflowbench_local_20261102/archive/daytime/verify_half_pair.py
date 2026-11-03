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
from task4_half_pair_stress import g


def main():
    name = 'task4_half_pair_stress'
    out = d.OUT/name
    rows = pd.read_csv(out/'all_cases.csv')
    assert len(rows)==160 and rows.case.nunique()==40
    errors, scale_errors, stationarity_errors = [], [], []
    for panel, group in rows.groupby('panel'):
        paths,A,_,_,_ = base.load(panel,'train')
        X = g.family.structure.features(paths)['span_origin_destination']
        dest = pd.factorize(paths.destination_zone,sort=True)[0]
        for case, actions in group.groupby('case'):
            key = hashlib.sha256(case.encode()).hexdigest()[:16]
            file = d.WORK/'task4_half_pair_cases'/key/'case.npz'
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
                for k in range(5):
                    train = z['fold'] != k
                    pair = design[train,-1]
                    rms = max(float(np.linalg.norm(pair)/np.sqrt(train.sum())),1e-12)
                    scale_errors.append(abs(rms-z['pair_scales'][k])/rms)
                    normalized = design[train].copy()
                    normalized[:,-1] /= z['pair_scales'][k]
                    target = prior[2,train]/z['scales'][k]
                    beta = z['half005_coef'][k]
                    partial = z['partial_information'][k]
                    u = normalized[:,-1]
                    features = normalized[:,:-1]
                    np.testing.assert_allclose(partial,u @ (u-features@g.qr.ridge_qr(features,u)),rtol=1e-12)
                    np.testing.assert_allclose(beta[-1],z['unit005_coef'][k,-1]*(partial+1)/(2*partial),rtol=1e-12,atol=1e-12)
                    assert z['geometry005_coef'][k,-1]==0.
                    penalty = beta.copy()
                    penalty[0] = 0.
                    penalty[-1] *= partial
                    gradient = normalized.T@(normalized@beta-target)+penalty
                    stationarity_errors.append(float(np.max(np.abs(gradient))/max(1.,np.max(np.abs(normalized.T@target)))))
                for row in actions.itertuples():
                    od,total = score(z[row.method],z['truth'][2],z['prior'][2],A,observed[2],dest)
                    errors.extend([abs(od-row.synthetic_S_od),abs(total-row.synthetic_S_ODME)])
    d.save(out/'independent_readback.json',dict(actions=160,fold_checks=200,
        max_relative_scale_error=max(scale_errors), max_relative_stationarity_error=max(stationarity_errors),
        max_score_error=max(errors), source_sha256=d.sha(__file__)))
    assert max(errors)<1e-12 and max(scale_errors)<1e-12 and max(stationarity_errors)<1e-9
    repeat = d.WORK/'publication_smoke/half_pair'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_half_pair.py'
    d.state(status='running',stage='half_pair_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['all_cases.csv','summary.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/file),pd.read_csv(repeat/name/file),check_exact=True)
    assert (out/'failures.json').read_bytes()==(repeat/name/'failures.json').read_bytes()
    for case, group in rows.groupby('case'):
        key = hashlib.sha256(case.encode()).hexdigest()[:16]
        assert set(group.artifact_sha256)=={d.sha(repeat/'work/task4_half_pair_cases'/key/'case.npz')}
    d.save(out/'portable_reproduction.json',dict(cases=40,matching_artifact_hashes=40,exact_csv_match=True,entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='half_pair_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
