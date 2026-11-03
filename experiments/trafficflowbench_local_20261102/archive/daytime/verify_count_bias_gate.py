"""Read back all gate decisions and scores, then reproduce with the portable runner."""
import hashlib
import msvcrt
import subprocess
import sys

import numpy as np
import pandas as pd

import common as d
import task4_span_shrink_stress as base
from verify_cycle_budget import score


def main():
    out = d.OUT / 'task4_count_bias_gate'
    rows = pd.read_csv(out / 'all_cases.csv')
    assert len(rows) == 480 and rows.case.nunique() == 160
    hashes, errors, real_biases = {}, [], []
    for panel, group in rows.groupby('panel'):
        paths, A, _, _, _ = base.load(panel, 'train')
        dest = pd.factorize(paths.destination_zone, sort=True)[0]
        for case, actions in group.groupby('case'):
            key = hashlib.sha256(case.encode()).hexdigest()[:16]
            folder = d.WORK / 'task4_count_bias_cases' / key
            for filename in ['inputs.npz', 'predictions.npz']:
                hashes[f'{key}/{filename}'] = d.sha(folder / filename)
            assert set(actions.input_sha256) == {hashes[f'{key}/inputs.npz']}
            with np.load(folder / 'inputs.npz') as z, np.load(folder / 'predictions.npz') as p:
                priors, counts = z['prior'], z['counts']
                bias = (A.sum(axis=0) @ priors.T) / counts.sum(axis=1) - 1
                enabled = bool(np.max(bias) <= .1)
                assert np.allclose(bias, z['bias'], atol=1e-12)
                assert enabled == bool(z['gate_enabled']) and (actions.gate_enabled == enabled).all()
                assert np.array_equal(p['bias_gated005'], p['raw005' if enabled else 'baseline'])
                for action in actions.itertuples():
                    od, total = score(p[action.method], z['truth'][2], priors[2], A, counts[2], dest)
                    errors.extend([abs(od-action.synthetic_S_od), abs(total-action.synthetic_S_ODME)])
        panel_bias = []
        for split in ['train', 'validation', 'private']:
            _, matrix, count, prior, _ = base.load(panel, split)
            panel_bias.append(float(matrix.sum(axis=0) @ prior / count.sum() - 1))
        real_biases.append(dict(panel=panel, **dict(zip(['train', 'validation', 'private'], panel_bias)), gate_enabled=max(panel_bias) <= .1))
    assert max(errors) < 1e-12
    pd.DataFrame(real_biases).to_csv(out / 'public_gate_activation.csv', index=False)
    d.save(out / 'independent_readback.json', dict(actions=480, max_error=max(errors), artifact_hashes=hashes, source_sha256=d.sha(__file__)))
    repeat = d.WORK / 'publication_smoke/count_bias_gate'
    runner = d.WORK / 'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_count_bias_gate.py'
    d.state(status='running', stage='count_bias_gate_reproduction')
    subprocess.run([sys.executable, '-X', 'utf8', str(runner), '--release-root', str(d.DATA), '--output', str(repeat)], check=True)
    for name in ['all_cases.csv', 'by_noise.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/name), pd.read_csv(repeat/'stress'/name), check_exact=True)
    for rel, digest in hashes.items():
        assert d.sha(repeat/'work/task4_count_bias_cases'/rel) == digest, rel
    d.save(out/'portable_reproduction.json', dict(cases=160, exact_csv_match=True, matching_artifacts=len(hashes), entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete', stage='count_bias_gate_verified')


if __name__ == '__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
