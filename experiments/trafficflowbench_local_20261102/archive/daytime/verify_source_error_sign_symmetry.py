import hashlib

import numpy as np
import pandas as pd

import common as d
from task4_exact_pair_audit import base as inputs
from verify_cycle_budget import score


def main():
    folder = d.OUT / 'task4_source_error_sign_symmetry'
    rows = pd.read_csv(folder / 'by_case.csv', dtype={'seed': 'int64'})
    assert len(rows) == 160 and rows.case.nunique() == 80
    contrasts = []
    for panel, group in rows.groupby('panel'):
        paths, A, *_ = inputs.load(panel, 'train')
        origin, dest = [paths[k].str[1:].astype(int).to_numpy() for k in ['origin_zone', 'destination_zone']]
        for case, table in group.groupby('case'):
            name = hashlib.sha256(case.encode()).hexdigest()[:16] + '.npz'
            result = d.WORK / 'task4_source_error_sign_symmetry_cases' / name
            assert len(table) == 2 and (table.artifact_sha256 == d.sha(result)).all()
            with np.load(result) as z, np.load(d.WORK / 'task4_source_agreement_stress_cases' / name) as ref:
                base, truth, count, priors = [ref[k] for k in ['baseline', 'truth', 'counts', 'priors']]
                b = priors[2]
                delta = z['source_change']
                assert (priors[:2] - np.abs(delta)).min() >= -1e-8
                for row in table.itertuples():
                    assert int(z['seed']) == row.seed
                    movement = np.abs(delta).sum() / min(priors[:2].sum(axis=1))
                    assert movement <= .1 + 1e-12
                    np.testing.assert_allclose(movement, row.source_movement, atol=1e-12, rtol=0)
                    scores = {}
                    for scenario in ('clean', 'aligned_plus', 'aligned_minus', 'opposed_plus', 'opposed_minus'):
                        flow = z[f'{row.rule}_{scenario}']
                        change = flow - base
                        for vector in (delta, change):
                            assert np.max(np.abs(A @ vector)) < 1e-7
                            for idx in (origin, dest):
                                assert np.max(np.abs(np.bincount(idx, weights=vector))) < 1e-7
                        assert flow.min() >= -1e-8
                        assert np.max(np.abs(change) - .05 * base) < 1e-7
                        np.testing.assert_allclose(np.abs(flow-b).sum(), np.abs(base-b).sum(), atol=1e-7, rtol=0)
                        np.testing.assert_allclose(np.abs(change).sum()/base.sum(), row.target_movement, atol=1e-12, rtol=0)
                        scores[scenario] = score(flow, truth, b, A, count, dest)[0]
                    aligned = (scores['aligned_plus'] + scores['aligned_minus']) / 2
                    opposed = (scores['opposed_plus'] + scores['opposed_minus']) / 2
                    contrasts.append(dict(case=case, panel=panel, noise=row.noise, rule=row.rule,
                                          first_positive=scores['aligned_plus']-scores['opposed_plus'],
                                          first_negative=scores['aligned_minus']-scores['opposed_minus'],
                                          symmetric=aligned-opposed))
                    actual = [aligned-scores['clean'], opposed-scores['clean'],
                              aligned-opposed]
                    np.testing.assert_allclose(actual, [row.aligned_effect, row.opposed_effect, row.aligned_minus_opposed],
                                               atol=1e-12, rtol=0)
    pd.DataFrame(contrasts).groupby(['noise', 'rule'])[['first_positive', 'first_negative', 'symmetric']].mean().to_csv(folder / 'matched_budget_sign_contrasts.csv')
    print('Verified 80 artifacts and 800 fitted flows')


if __name__ == '__main__':
    main()
