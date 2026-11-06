"""Exploratory probability calibration on exposed synthetic Task4 cases."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.cases)
    scores = cases.pivot(index='case', columns='method', values='synthetic_S_od')
    frame = cases[cases.method.eq('pair005')].set_index('case').join(
        (scores.pair005 - scores.geometry005).rename('delta'))
    assert len(frame) == 960 and frame.solver_success.all()
    feature = frame.pair_prior_prediction_MSE_reduction.clip(-1, 1).to_numpy().reshape(-1, 1)
    win = frame.delta.gt(1e-12).to_numpy()
    rows = []
    for split in ('panel', 'noise'):
        for group in sorted(frame[split].unique()):
            test = frame[split].eq(group).to_numpy()
            model = LogisticRegression(C=1).fit(feature[~test], win[~test])
            probability = model.predict_proba(feature[test])[:, 1]
            prior = win[~test].mean()
            chosen = probability >= .65
            delta = frame.delta.to_numpy()[test]
            rows.append(dict(split=split, group=group, cases=test.sum(), wins=win[test].sum(),
                             mean_probability=probability.mean(), selected=chosen.sum(),
                             selected_losses=(chosen & (delta < -1e-12)).sum(),
                             gain_sum=delta[chosen].sum(),
                             always_pair_sum=delta.sum(),
                             brier_sum=np.square(probability - win[test]).sum(),
                             base_brier_sum=np.square(prior - win[test]).sum()))
    result = pd.DataFrame(rows)
    result.to_csv(args.output / 'by_group.csv', index=False)
    result.groupby('split').sum(numeric_only=True).drop(columns='mean_probability').to_csv(
        args.output / 'summary.csv')
    metadata = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in [('input_sha256', args.cases), ('source_sha256', Path(__file__))]}
    metadata.update(scope='Previously exposed synthetic cases only', threshold=.65, draws=0)
    (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
