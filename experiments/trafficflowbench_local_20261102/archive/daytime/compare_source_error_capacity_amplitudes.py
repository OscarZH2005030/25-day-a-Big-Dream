"""Compare full and quarter source amplitudes at an identical target budget."""
import hashlib

import numpy as np
import pandas as pd

import common as d
from task4_exact_pair_audit import base as inputs
from verify_cycle_budget import score


def main():
    names = ('task4_source_error_symmetric_capacity', 'task4_source_error_quarter_capacity')
    tables = [pd.read_csv(d.OUT / study / 'by_case.csv', dtype={'seed':'int64'})
              .query("rule == 'mean_source'").set_index('case') for study in names]
    assert set(tables[0].index) == set(tables[1].index)
    rows = []
    for panel, group in tables[0].groupby('panel'):
        paths, A, *_ = inputs.load(panel, 'train')
        dest = paths.destination_zone.str[1:].astype(int).to_numpy()
        for case, row in group.iterrows():
            name = hashlib.sha256(case.encode()).hexdigest()[:16] + '.npz'
            budget = min(row.target_movement, tables[1].loc[case].target_movement)
            vectors = []
            records = []
            with np.load(d.WORK / 'task4_source_agreement_stress_cases' / name) as ref:
                base, truth, count, priors = [ref[k] for k in ('baseline','truth','counts','priors')]
                for study, table in zip(names, tables):
                    artifact = d.WORK / (study + '_cases') / name
                    assert d.sha(artifact) == table.loc[case].artifact_sha256
                    with np.load(artifact) as z:
                        vectors.append(z['source_change'])
                        for rule in ('agreement_source','mean_source'):
                            scores = {}
                            for scenario in ('clean','aligned_plus','aligned_minus','opposed_plus','opposed_minus'):
                                change = z[f'{rule}_{scenario}'] - base
                                factor = budget * base.sum() / max(1e-12, np.abs(change).sum())
                                assert factor <= 1 + 1e-8
                                flow = base + factor * change
                                scores[scenario] = score(flow, truth, priors[2], A, count, dest)[0]
                            aligned = (scores['aligned_plus']+scores['aligned_minus'])/2
                            opposed = (scores['opposed_plus']+scores['opposed_minus'])/2
                            records.append(dict(case=case, panel=panel, noise=row.noise, rule=rule, study=study,
                                                target_movement=budget, contrast=aligned-opposed))
                np.testing.assert_array_equal(vectors[0]*.25, vectors[1])
            rows.extend(records)
    frame = pd.DataFrame(rows)
    out = d.OUT / names[1]
    frame.to_csv(out / 'matched_amplitude_by_case.csv', index=False)
    wide = frame.pivot(index=['case','noise','rule'], columns='study', values='contrast')
    wide['full_minus_quarter'] = wide[names[0]] - wide[names[1]]
    wide.groupby(level=['noise','rule']).mean().to_csv(out / 'matched_amplitude_summary.csv')


if __name__ == '__main__':
    main()
