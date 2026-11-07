"""Summarize paired selection differences within each covariance experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('span', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    keys = ['panel', 'selection', 'exponent', 'law']
    metrics = ['total_reduction', 'common_reduction', 'origin_reduction', 'destination_reduction']
    provenance = {}
    paired = []
    for label, folder in [('origin_destination', args.original), ('with_span', args.span)]:
        assert json.loads((folder / 'failures.json').read_text(encoding='utf-8')) == []
        rows = pd.read_csv(folder / 'by_panel.csv')
        assert set(rows.design) == {'other', 'total'}
        other = rows.loc[rows.design.eq('other')].set_index(keys).sort_index()
        total = rows.loc[rows.design.eq('total')].set_index(keys).sort_index()
        assert other.index.is_unique and total.index.is_unique and other.index.equals(total.index)
        differences = (other[metrics] - total[metrics]).reset_index()
        differences.insert(0, 'covariance_experiment', label)
        paired.append(differences)
        provenance[label] = {filename: digest(folder / filename)
                             for filename in ('by_panel.csv', 'protocol.json', 'input_sha256.json')}
    assert provenance['origin_destination']['input_sha256.json'] == provenance['with_span']['input_sha256.json']
    assert paired[0][keys].equals(paired[1][keys])
    differences = pd.concat(paired, ignore_index=True)
    differences['common_positive'] = differences.common_reduction.gt(1e-12)
    differences['common_negative'] = differences.common_reduction.lt(-1e-12)
    groups = ['covariance_experiment', 'selection', 'exponent', 'law']
    summary = differences.groupby(groups)[metrics].mean().join(
        differences.groupby(groups)[['common_positive', 'common_negative']].sum())
    args.output.mkdir(parents=True, exist_ok=True)
    differences.to_csv(args.output / 'by_panel.csv', index=False)
    summary.to_csv(args.output / 'summary.csv')
    metadata = {'source_sha256': digest(Path(__file__)), 'inputs': provenance,
                'scope': 'Other-target minus total-target inside each experiment. Each experiment defines its own common unmeasured OD set. Covariance laws are hypothetical, and cell-set changes prevent direct comparison of absolute common-set reductions across experiments.'}
    (args.output / 'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(summary.xs('balanced', level='selection').to_string())


if __name__ == '__main__':
    main()
