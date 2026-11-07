"""Audit public count fit of the active Task4 candidate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import task4_span_shrink_stress as base


if __name__ == '__main__':
    folder = Path('work/daytime/jayden-task4-publish/experiments/trafficflowbench_local_20261102/task4')
    old_path = folder / 'frozen_reference_path_flow.csv'
    new_path = folder / 'forward_candidate_path_flow.csv'
    old, new = [pd.read_csv(path) for path in (old_path, new_path)]
    keys = ['panel','departure_time','path_id','origin_zone','destination_zone']
    pd.testing.assert_frame_equal(old[keys], new[keys])
    rows = []
    hashes = {}
    for (panel, time), group in old.groupby(['panel', 'departure_time']):
        split = {'SYN_PM_VALIDATION':'validation', 'SYN_PM_PRIVATE':'private'}[time]
        paths, matrix, counts, prior, provenance = base.load(panel, split)
        hashes.update(provenance)
        first = group.set_index('path_id').loc[paths.path_id].path_flow.to_numpy()
        second = new.loc[group.index].set_index('path_id').loc[paths.path_id].path_flow.to_numpy()
        rows.append(dict(panel=panel, split=split,
                         baseline_count_nmae=np.abs(matrix@first-counts).sum()/counts.sum(),
                         candidate_count_nmae=np.abs(matrix@second-counts).sum()/counts.sum(),
                         maximum_count_prediction_change=np.abs(matrix@(second-first)).max(),
                         prior_l1_change=abs(np.abs(second-prior).sum()-np.abs(first-prior).sum())))
    out = Path('outputs/daytime/task4_active_candidate_public_audit')
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'by_road.csv',index=False)
    frame.groupby('split').mean(numeric_only=True).to_csv(out/'summary.csv')
    (out/'sha256.json').write_text(json.dumps(dict(inputs=hashes,
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        candidate=hashlib.sha256(new_path.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(frame.groupby('split')[['baseline_count_nmae','candidate_count_nmae']].mean())
