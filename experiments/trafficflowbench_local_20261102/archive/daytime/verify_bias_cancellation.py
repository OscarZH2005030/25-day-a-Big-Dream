"""Independently verify the paired cancellation control and reproduce it."""
import msvcrt
import subprocess
import sys

import numpy as np
import pandas as pd

import common as d
import task4_span_shrink_stress as base
from verify_cycle_budget import score


def main():
    name = 'task4_bias_cancellation_stress'
    out = d.OUT/name
    rows = pd.read_csv(out/'all_cases.csv')
    assert len(rows) == 240 and rows.case.nunique() == 40
    digests, errors = {}, []
    for panel, panel_rows in rows.groupby('panel'):
        paths, A, _, _, _ = base.load(panel, 'train')
        dest = pd.factorize(paths.destination_zone, sort=True)[0]
        for family, pair in panel_rows.groupby('family'):
            folder = d.WORK/'task4_bias_cancellation_cases'/panel/family
            with np.load(folder/'original.npz') as original, np.load(folder/'bulk_normalized.npz') as normalized:
                for key in ['truth', 'counts', 'fold', 'seed']:
                    np.testing.assert_array_equal(original[key], normalized[key])
                prior, counts = original['prior'], original['counts']
                expected = prior*(counts.sum(axis=1)/(A.sum(axis=0) @ prior.T))[:, None]
                np.testing.assert_allclose(expected, normalized['prior'], rtol=1e-13)
                for variant, z in [('original', original), ('bulk_normalized', normalized)]:
                    rel = f'{panel}/{family}/{variant}.npz'
                    digests[rel] = d.sha(folder/f'{variant}.npz')
                    bias = (A.sum(axis=0) @ z['prior'].T)/counts.sum(axis=1)-1
                    admitted = bool(bias.max() <= .1)
                    actions = pair[pair.variant==variant]
                    assert set(actions.artifact_sha256)=={digests[rel]}
                    assert (actions.gate_enabled==admitted).all()
                    flows = dict(baseline=z['baseline'], raw005=z['direct'], gated005=z['direct' if admitted else 'baseline'])
                    for row in actions.itertuples():
                        od, total = score(flows[row.method], z['truth'][2], z['prior'][2], A, counts[2], dest)
                        errors.extend([abs(od-row.synthetic_S_od), abs(total-row.synthetic_S_ODME)])
    assert max(errors)<1e-12
    d.save(out/'independent_readback.json', dict(actions=len(rows), max_score_error=max(errors), artifact_hashes=digests, source_sha256=d.sha(__file__)))
    repeat = d.WORK/'publication_smoke/bias_cancellation'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_bias_cancellation.py'
    d.state(status='running', stage='bias_cancellation_reproduction')
    if '--check-existing' not in sys.argv:
        subprocess.run([sys.executable, '-X', 'utf8', str(runner), '--release-root', str(d.DATA), '--output', str(repeat)], check=True)
    score_errors = {}
    for filename in ['all_cases.csv','summary.csv']:
        left, right = [pd.read_csv(path) for path in [out/filename, repeat/name/filename]]
        for column in left.select_dtypes(include='floating'):
            score_errors[f'{filename}/{column}'] = float((left[column]-right[column]).abs().max())
        pd.testing.assert_frame_equal(left.drop(columns='artifact_sha256', errors='ignore'),
                                      right.drop(columns='artifact_sha256', errors='ignore'),
                                      check_exact=False, rtol=0, atol=1e-12)
    matching = 0
    maximum_relative_errors = {}
    for rel, digest in digests.items():
        repeated_file = repeat/'work/task4_bias_cancellation_cases'/rel
        matching += int(d.sha(repeated_file)==digest)
        with np.load(d.WORK/'task4_bias_cancellation_cases'/rel) as left, np.load(repeated_file) as right:
            assert left.files == right.files
            for field in left.files:
                a, b = left[field], right[field]
                if field in ['fold','seed']:
                    np.testing.assert_array_equal(a,b)
                else:
                    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
                    error = float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(a)))))
                    maximum_relative_errors[field] = max(maximum_relative_errors.get(field,0.),error)
    d.save(out/'portable_reproduction.json', dict(paired_cases=40, initial_exact_comparison_failed=True,
        matching_artifact_bytes=matching, artifacts_numerically_verified=len(digests),
        maximum_absolute_table_errors=score_errors, maximum_relative_array_errors=maximum_relative_errors,
        exact_generator_folds_and_seeds=True, entrypoint_sha256=d.sha(runner),
        source_sha256=d.sha(__file__)))
    d.state(status='stage_complete', stage='bias_cancellation_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
