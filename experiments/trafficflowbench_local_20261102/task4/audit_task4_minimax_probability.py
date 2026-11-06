"""Minimax allocation across synthetic prior-error and drift families."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.cases)
    methods = ['baseline', 'geometry005', 'pair005', 'pair025', 'pair100']
    scores = cases.pivot(index=['noise', 'drift', 'case'], columns='method', values='synthetic_S_od')[methods]
    assert len(scores) == 960 and np.isfinite(scores).all().all()
    gains = scores.subtract(scores.baseline, axis=0)
    rows = []
    for held_out in ['none', *sorted(gains.index.unique('noise'))]:
        train = gains if held_out == 'none' else gains[gains.index.get_level_values('noise') != held_out]
        means = train.groupby(level=['noise', 'drift']).mean()
        result = linprog([0.] * len(methods) + [-1.],
                         A_ub=np.c_[-means.to_numpy(), np.ones(len(means))], b_ub=np.zeros(len(means)),
                         A_eq=[[1.] * len(methods) + [0.]], b_eq=[1.],
                         bounds=[(0., 1.)] * len(methods) + [(None, None)], method='highs')
        if not result.success:
            rows.append(dict(held_out=held_out, failure=result.message))
            continue
        weights = result.x[:-1]
        assert abs(weights.sum() - 1.) < 1e-8
        assert abs((means @ weights).min() - result.x[-1]) < 1e-8
        test = gains if held_out == 'none' else gains[gains.index.get_level_values('noise') == held_out]
        evaluated = test.groupby(level=['noise', 'drift']).mean()
        for (noise, drift), group in evaluated.iterrows():
            rows.append(dict(held_out=held_out, noise=noise, drift=drift,
                             training_worst_gain=result.x[-1], expected_gain=group @ weights,
                             **dict(zip(methods, weights))))
    pd.DataFrame(rows).to_csv(args.output / 'by_family.csv', index=False)
    means = gains.groupby(level=['noise', 'drift']).mean()
    frontier = []
    for allowance in (0., .001, .005):
        result = linprog([0., -1., -1., -1., -1.], A_ub=-means.to_numpy(),
                         b_ub=np.full(len(means), allowance),
                         A_eq=[np.ones(len(methods))], b_eq=[1.], bounds=(0., 1.), method='highs')
        if not result.success:
            frontier.append(dict(allowance=allowance, failure=result.message))
            continue
        weights = result.x
        worst = (means @ weights).min()
        assert worst >= -allowance - 1e-8
        frontier.append(dict(allowance=allowance, worst_gain=worst,
                             mean_gain=gains.mean() @ weights, **dict(zip(methods, weights))))
    pd.DataFrame(frontier).to_csv(args.output / 'risk_frontier.csv', index=False)
    info = dict(input_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exposed synthetic cases. Expected scores of randomized method choices, not flow mixtures.',
                seed=None)
    (args.output / 'protocol.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
