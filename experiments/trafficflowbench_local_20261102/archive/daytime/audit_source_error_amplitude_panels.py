"""Check whether the matched-amplitude contrast depends on a single road panel."""
import hashlib

import numpy as np
import pandas as pd

import common as d


def main():
    source = d.OUT / 'task4_source_error_quarter_capacity/matched_amplitude_by_case.csv'
    out = d.OUT / 'task4_source_error_amplitude_panel_check'
    seed = 2026110353
    d.save(out / 'protocol.json', dict(seed=seed, draws=10000, input_sha256=d.sha(source),
        code_sha256=d.sha(__file__), scope='Posthoc descriptive sensitivity on reused artificial cases.',
        design='Pair by case and use whole-road panels as resampling blocks. Use common draws across noise families.'))
    rows = pd.read_csv(source)
    assert not rows.duplicated(['case','panel','noise','rule','study']).any()
    paired = rows.pivot(index=['case','panel','noise','rule'], columns='study', values='contrast')
    assert len(paired) == 160 and not paired.isna().any().any()
    paired['delta'] = paired.task4_source_error_symmetric_capacity - paired.task4_source_error_quarter_capacity
    panel = paired.groupby(['panel','noise','rule']).delta.mean().unstack(['noise','rule'])
    assert panel.shape == (10, 8) and not panel.isna().any().any()
    values = panel.to_numpy()
    sampled = np.random.default_rng(seed).integers(0,len(panel),size=(10000,len(panel)))
    means = values[sampled].mean(axis=1)
    omitted = (values.sum(axis=0) - values) / (len(panel)-1)
    out.mkdir(parents=True, exist_ok=True)
    panel.stack(['noise','rule']).rename('delta').to_csv(out / 'by_panel.csv')
    pd.DataFrame(omitted,index=panel.index,columns=panel.columns).stack(['noise','rule']).rename('delta').to_csv(out / 'leave_one_panel_out.csv')
    summary = pd.DataFrame(dict(mean=values.mean(axis=0), loo_min=omitted.min(axis=0), loo_max=omitted.max(axis=0),
        resample_q025=np.quantile(means,.025,axis=0),resample_q975=np.quantile(means,.975,axis=0)),index=panel.columns)
    summary.to_csv(out / 'summary.csv')
    d.save(out / 'failures.json', [])
    d.state(status='stage_complete', stage='task4_source_error_amplitude_panel_check_complete',
            next='Review descriptive road-panel sensitivity before choosing the next stress test.')
    print(summary.to_string())


if __name__ == '__main__':
    main()
