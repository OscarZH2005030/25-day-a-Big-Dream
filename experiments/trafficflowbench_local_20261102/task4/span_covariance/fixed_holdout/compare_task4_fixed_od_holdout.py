"""Audit and compare fixed reserved-cell OD measurement designs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHTS = {'zero': 0., 'weak': .03125, 'strong': .125}
KEYS = ['panel', 'selection', 'exponent', 'law']


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--weight', choices=WEIGHTS, help='Audit one completed stage while the queue runs')
    parser.add_argument('--portable-layout', action='store_true', help='Read zero/weak/strong release CLI outputs')
    args = parser.parse_args()
    weights = {args.weight: WEIGHTS[args.weight]} if args.weight else WEIGHTS
    rows, provenance = [], {}
    baseline_hashes = None
    baseline_mask = None
    for name, weight in weights.items():
        folder = (args.output_root / name / 'task4_fixed_od_holdout' if args.portable_layout else
                  args.output_root / f'task4_fixed_od_holdout_{name}')
        protocol = load_json(folder / 'protocol.json')
        assert protocol['shared_weights']['span'] == weight
        assert protocol['fixed_holdout']['seed'] == 2026110707
        assert load_json(folder / 'failures.json') == []
        hashes = load_json(folder / 'input_sha256.json')
        masks = load_json(folder / 'holdout_sha256.json')
        if baseline_hashes is None:
            baseline_hashes, baseline_mask = hashes, masks
        assert hashes == baseline_hashes and masks == baseline_mask
        frame = pd.read_csv(folder / 'by_panel.csv')
        assert set(frame.design) == {'total', 'other', 'heldout'}
        assert len(frame) == 900 and set(frame.panel) == set(masks)
        assert np.isfinite(frame.select_dtypes('number').to_numpy()).all()
        values = frame.pivot(index=KEYS, columns='design', values='heldout_relative_reduction')
        assert values.notna().all().all()
        for design in ('other', 'heldout'):
            paired = values[[design, 'total']].rename(columns={design: 'target_reduction'}).reset_index()
            paired['objective'] = design
            paired['span_share'] = weight
            paired['gain'] = paired.target_reduction - paired.total
            paired['win'] = paired.gain.gt(1e-12)
            paired['loss'] = paired.gain.lt(-1e-12)
            rows.append(paired)
        provenance[name] = {key: hashlib.sha256((folder / key).read_bytes()).hexdigest()
                            for key in ('protocol.json', 'by_panel.csv', 'input_sha256.json', 'holdout_sha256.json')}
    result = pd.concat(rows, ignore_index=True)
    keys = ['span_share', 'selection', 'exponent', 'law', 'objective']
    summary = result.groupby(keys).agg(mean_gain=('gain', 'mean'), wins=('win', 'sum'), losses=('loss', 'sum'))
    out = args.output_root / ('task4_fixed_od_holdout_paired' + (f'_{args.weight}' if args.weight else ''))
    out.mkdir(exist_ok=True)
    result.to_csv(out / 'by_panel.csv', index=False)
    summary.to_csv(out / 'summary.csv')
    (out / 'provenance.json').write_text(json.dumps({
        'inputs': provenance, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Fixed reserved OD cells are identical across policies and covariance assumptions. Gains are within-law relative variance reductions, not OD recovery scores.'},
        indent=2), encoding='utf-8')
    print(summary.xs('balanced', level='selection').to_string())


if __name__ == '__main__':
    main()
