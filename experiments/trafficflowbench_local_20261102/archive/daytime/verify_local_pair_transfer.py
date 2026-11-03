"""Read back saved weak-prior fits and reproduce the complete portable audit."""
import json
import msvcrt
import subprocess
import sys

import common as d
import numpy as np
import pandas as pd
from task4_local_pair_transfer import exact, source_features


def main():
    name = 'task4_local_pair_transfer'
    out = d.OUT/name
    rows = pd.read_csv(out/'by_panel_fold.csv')
    folds = pd.read_csv(out/'fold_assignment.csv')
    hashes = json.loads((out/'model_sha256.json').read_text())
    errors = []
    for panel,group in rows.groupby('panel'):
        priors = {split:exact.base.load(panel,split) for split in exact.SPLITS}
        paths = priors['train'][0]
        X = exact.structure.features(paths)['span_origin_destination']
        Q,_ = exact.basis(X)
        fold = folds[folds.panel==panel].set_index('path_id').loc[paths.path_id,'fold'].to_numpy()
        for split,actions in group.groupby('split'):
            source = np.mean([p[3]/max(float(np.median(p[3][p[3]>0])),1e-9)
                              for key,p in priors.items() if key!=split],axis=0)
            features,_ = source_features(paths,X,Q,source)
            for row in actions.itertuples():
                rel = f'task4_local_pair_models/{panel}/{split}/{row.method}/fold{row.fold}.npz'
                assert d.sha(d.WORK/rel)==hashes[rel]
                with np.load(d.WORK/rel) as z:
                    test = fold==row.fold
                    assert row.n_test==int(test.sum()) and int(z['fold'])==row.fold
                    residual = np.maximum(features[row.method][test]@z['coef'],0)-priors[split][3][test]/z['scale']
                    errors.extend([abs(float(residual@residual/test.sum())-row.mse),
                                   abs(float(np.abs(residual).sum()/test.sum())-row.mae)])
    assert len(rows)==600 and len(hashes)==600 and max(errors)<1e-10
    d.save(out/'independent_readback.json',dict(actions=600,max_score_error=max(errors),source_sha256=d.sha(__file__)))
    repeat = d.WORK/'publication_smoke/local_pair_transfer'
    runner = d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_local_pair_transfer.py'
    d.state(status='running',stage='local_pair_transfer_reproduction')
    subprocess.run([sys.executable,'-X','utf8',str(runner),'--release-root',str(d.DATA),'--output',str(repeat)],check=True)
    for file in ['by_panel_fold.csv','paired_mse_improvement.csv','summary.csv','fold_assignment.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/file),pd.read_csv(repeat/name/file),check_exact=True)
    for rel,digest in hashes.items():
        assert d.sha(repeat/'work'/rel)==digest,rel
    d.save(out/'portable_reproduction.json',dict(exact_csv_match=True,matching_model_hashes=len(hashes),entrypoint_sha256=d.sha(runner)))
    d.state(status='stage_complete',stage='local_pair_transfer_verified')


if __name__=='__main__':
    with (d.ROOT/'work/overnight/research.lock').open('r+b') as lock:
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        main()
