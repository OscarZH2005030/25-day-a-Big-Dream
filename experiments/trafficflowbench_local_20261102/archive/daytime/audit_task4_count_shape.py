"""Describe scale-invariant count residuals on completed development cases."""
import hashlib
import msvcrt

import common as d
import task4_span_shrink_stress as base
import numpy as np
import pandas as pd


def shape_tv(loaded, observed):
    return float(.5*np.abs(loaded/loaded.sum()-observed/observed.sum()).sum())


def main():
    out = d.OUT/'task4_count_shape_audit'
    d.save(out/'protocol.json', dict(scope='Descriptive analysis of reused synthetic cases and public inputs. No thresholds, labels for real OD, or candidate changes.',
                                    metric='Max across inputs of TV between normalized loaded and observed link count vectors',
                                    source_sha256=d.sha(__file__)))
    rows, hashes = [], {}
    cases = pd.read_csv(d.OUT/'task4_count_bias_gate/all_cases.csv').query("method == 'baseline'")
    pairs = pd.read_csv(d.OUT/'task4_bias_cancellation_stress/all_cases.csv').query("method == 'baseline'")
    for panel, group in cases.groupby('panel'):
        _, A, _, _, inputs = base.load(panel,'train')
        hashes.update(inputs)
        for row in group.itertuples():
            key = hashlib.sha256(row.case.encode()).hexdigest()[:16]
            with np.load(d.WORK/'task4_count_bias_cases'/key/'inputs.npz') as z:
                value = max(shape_tv(A@b,c) for b,c in zip(z['prior'],z['counts']))
            rows.append(dict(panel=panel, case=row.case, group=row.noise, max_shape_tv=value))
        for row in pairs[pairs.panel==panel].itertuples():
            with np.load(d.WORK/'task4_bias_cancellation_cases'/panel/row.family/f'{row.variant}.npz') as z:
                value = max(shape_tv(A@b,c) for b,c in zip(z['prior'],z['counts']))
            rows.append(dict(panel=panel, case=row.case, group='paired_'+row.variant, max_shape_tv=value))
        values = []
        for split in ['train','validation','private']:
            _, matrix, count, prior, inputs = base.load(panel,split)
            hashes.update(inputs)
            values.append(shape_tv(matrix@prior,count))
        rows.append(dict(panel=panel, case=panel, group='public_inputs', max_shape_tv=max(values)))
    frame = pd.DataFrame(rows)
    pair = frame[frame.group.str.startswith('paired')].pivot(index='case',columns='group',values='max_shape_tv')
    assert np.max(np.abs(pair.paired_original-pair.paired_bulk_normalized))<1e-12
    frame.to_csv(out/'by_case.csv',index=False)
    frame.groupby('group').max_shape_tv.agg(['count','min','median','max']).to_csv(out/'summary.csv')
    d.save(out/'input_sha256.json',hashes)
    d.state(status='stage_complete',stage='count_shape_described',next='Assess overlap before designing any new registered scale-invariant test. Existing cases are development only.')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
