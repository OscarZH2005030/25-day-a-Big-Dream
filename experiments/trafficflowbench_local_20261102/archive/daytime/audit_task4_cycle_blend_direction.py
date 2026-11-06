"""Compare deterministic OD flow blends with randomized endpoint choice."""
import hashlib
import msvcrt

import numpy as np
import pandas as pd

import common as d


def main(source=None):
    source = source or d.OUT / 'task4_od_cycle_stress' / 'all_cases.csv'
    out = d.OUT / 'task4_cycle_blend_direction'
    rows = []
    hashes = {}
    for case, group in pd.read_csv(source).groupby('case'):
        meta = group.set_index('method')
        folder = d.WORK / 'task4_od_cycle_cases' / hashlib.sha256(case.encode()).hexdigest()[:16]
        input_path, pred_path = folder / 'inputs.npz', folder / 'predictions.npz'
        hashes[case] = [d.sha(input_path), d.sha(pred_path)]
        assert hashes[case] == meta.loc['baseline', ['input_artifact_sha256', 'prediction_artifact_sha256']].tolist()
        with np.load(input_path) as data, np.load(pred_path) as pred:
            truth = data['artificial_truth'][2]
            total = truth.sum()
            baseline = pred['baseline']
            def score(flow):
                return max(0., 1. - np.abs(flow - truth).sum() / total)
            base = score(baseline)
            for method in ('cycle005', 'cycle025'):
                candidate = pred[method]
                end = score(candidate)
                assert abs(end - meta.loc[method, 'synthetic_S_od']) < 1e-10
                error = baseline - truth
                movement = candidate - baseline
                assert base > 0.
                derivative = -np.where(error != 0., np.sign(error) * movement,
                                       np.abs(movement)).sum() / total
                # Both endpoints are above the clipping floor, so concavity holds throughout.
                assert end > 0.
                assert end - base <= derivative + 1e-10
                record = meta.loc[method]
                rows.append(dict(panel=record.panel, noise=record.noise, drift=record.drift,
                                 method=method, right_derivative=derivative))
    frame = pd.DataFrame(rows)
    assert len(frame) == 960
    out.mkdir(parents=True, exist_ok=True)
    frame.groupby(['panel', 'noise', 'drift', 'method']).agg(
        cases=('right_derivative', 'size'),
        mean_derivative=('right_derivative', 'mean'),
        min_derivative=('right_derivative', 'min')
    ).to_csv(out / 'by_joint_stratum.csv')
    d.save(out / 'protocol.json', dict(source_sha256=d.sha(__file__), case_table_sha256=d.sha(source),
                                      input_prediction_hashes=hashes, draws=0,
                                      scope='Exposed synthetic cases; fixed weights and fixed endpoint predictions.'))


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
