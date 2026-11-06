"""Compare deterministic OD flow blends with randomized endpoint choice."""
import hashlib
import msvcrt

import numpy as np
import pandas as pd

import common as d


def main(source=None):
    source = source or d.OUT / 'task4_od_cycle_stress' / 'all_cases.csv'
    out = d.OUT / 'task4_cycle_blend_joint'
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
                for weight in (.1, .25, .5, .75):
                    blend = score((1. - weight) * baseline + weight * candidate)
                    expected = (1. - weight) * base + weight * end
                    record = meta.loc[method]
                    rows.append(dict(panel=record.panel, noise=record.noise, drift=record.drift,
                                     method=method, weight=weight, gain=blend-base,
                                     random_gain=expected-base, blend_advantage=blend-expected))
    frame = pd.DataFrame(rows)
    assert len(frame) == 480 * 8
    out.mkdir(parents=True, exist_ok=True)
    for filename, groups in [('by_joint_stratum.csv', ['panel', 'noise', 'drift'])]:
        frame.groupby(groups + ['method', 'weight']).agg(
            cases=('gain', 'size'), mean_gain=('gain', 'mean'), min_gain=('gain', 'min'),
            harms=('gain', lambda values: (values < -1e-12).sum()),
            random_mean_gain=('random_gain', 'mean'), mean_blend_advantage=('blend_advantage', 'mean')
        ).to_csv(out / filename)
    d.save(out / 'protocol.json', dict(source_sha256=d.sha(__file__), case_table_sha256=d.sha(source),
                                      input_prediction_hashes=hashes, draws=0,
                                      scope='Exposed synthetic cases; fixed weights and fixed endpoint predictions.'))


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
