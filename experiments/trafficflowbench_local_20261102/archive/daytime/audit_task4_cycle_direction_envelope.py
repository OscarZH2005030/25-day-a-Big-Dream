"""Compare deterministic OD flow blends with randomized endpoint choice."""
import hashlib
import msvcrt

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import common as d


def main(source=None):
    source = source or d.OUT / 'task4_od_cycle_stress' / 'all_cases.csv'
    out = d.OUT / 'task4_cycle_direction_envelope'
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
            for ratio in (0., .25, .5, .75, 1.):
                candidate = (1. - ratio) * pred['cycle005'] + ratio * pred['cycle025']
                end = score(candidate)
                assert end + 1e-10 >= (1. - ratio) * meta.loc['cycle005', 'synthetic_S_od'] + ratio * meta.loc['cycle025', 'synthetic_S_od']
                error = baseline - truth
                movement = candidate - baseline
                assert base > 0.
                derivative = -np.where(error != 0., np.sign(error) * movement,
                                       np.abs(movement)).sum() / total
                direction_change = pred['cycle025'] - pred['cycle005']
                # A supergradient of the concave right derivative as a function of ratio.
                slope = -np.where(error != 0., np.sign(error) * direction_change,
                                  np.sign(movement) * direction_change).sum() / total
                # Both endpoints are above the clipping floor, so concavity holds throughout.
                assert end > 0.
                assert end - base <= derivative + 1e-10
                record = meta.loc['baseline']
                rows.append(dict(panel=record.panel, noise=record.noise, drift=record.drift,
                                 ratio=ratio, right_derivative=derivative, slope=slope,
                                 gain_at_point_one=score(.9 * baseline + .1 * candidate) - base))
    frame = pd.DataFrame(rows)
    assert len(frame) == 2400
    out.mkdir(parents=True, exist_ok=True)
    frame.groupby(['panel', 'noise', 'drift', 'ratio']).agg(
        cases=('right_derivative', 'size'),
        mean_derivative=('right_derivative', 'mean'),
        min_derivative=('right_derivative', 'min'), mean_gain=('gain_at_point_one', 'mean'), mean_slope=('slope', 'mean')
    ).to_csv(out / 'by_joint_stratum.csv')
    means = frame.groupby(['panel', 'noise', 'drift', 'ratio'])[['right_derivative', 'slope']].mean().reset_index()
    # Each tangent dominates the group derivative for all ratios.
    # Maximizing the minimum of all tangents gives an optimistic upper bound.
    intercept = means.right_derivative - means.slope * means.ratio
    result = linprog([0., -1.], A_ub=np.c_[-means.slope, np.ones(len(means))],
                     b_ub=intercept, bounds=[(0., 1.), (None, None)], method='highs')
    diagnostics = dict(success=bool(result.success), status=int(result.status))
    if result.success:
        diagnostics.update(optimistic_maximum_worst_derivative=float(result.x[1]),
                           ratio_at_bound=float(result.x[0]),
                           grid_maximum_worst_derivative=float(
                               means.groupby('ratio').right_derivative.min().max()))
    d.save(out / 'envelope.json', diagnostics)
    d.save(out / 'protocol.json', dict(source_sha256=d.sha(__file__), case_table_sha256=d.sha(source),
                                      input_prediction_hashes=hashes, draws=0,
                                      scope='Exposed synthetic cases; fixed weights and fixed endpoint predictions.'))


if __name__ == '__main__':
    with (d.ROOT / 'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
