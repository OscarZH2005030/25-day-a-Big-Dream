"""Verify old-best preservation and that the delivered correction changes only allowed flow rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from residual_core import ROOT,BASE,WORKSPACE,save,sha


def main():
    preserved=json.loads((ROOT/'preserved-v2-sha256.json').read_text())
    assert all(sha(Path(p))==v for p,v in preserved.items())
    routes=json.loads((ROOT/'frozen-routes.json').read_text())['routes']
    adopted={r['panel'] for r in routes if r['alpha']>0}
    original=BASE/'candidate-submission/submission.csv'
    candidate=ROOT/'candidate-submission/submission.csv'
    changes=0;rows=0;maximum_delta=0.
    if adopted:
        assert candidate.exists()
        keys=pd.read_csv(WORKSPACE/'work/trafficflowbench-data/kaggle_public/submission_key.csv',
            usecols=['submission_id','task','panel'],chunksize=250000)
        a=pd.read_csv(original,chunksize=250000);b=pd.read_csv(candidate,chunksize=250000)
        for old,new,key in zip(a,b,keys,strict=True):
            assert len(old)==len(new)==len(key)
            assert np.array_equal(old.submission_id,new.submission_id)
            assert np.array_equal(new.submission_id,key.submission_id)
            for c in ['task','speed_kmh','queue_pred','path_flow']:
                assert np.array_equal(old[c].to_numpy(),new[c].to_numpy()),c+' changed'
            changed=old.flow_vph.to_numpy()!=new.flow_vph.to_numpy()
            allowed=(key.task=='state')&key.panel.isin(adopted)
            assert not (changed&~allowed.to_numpy()).any()
            changes+=int(changed.sum());rows+=len(old)
            maximum_delta=max(maximum_delta,float(np.max(np.abs(new.flow_vph.to_numpy()-old.flow_vph.to_numpy()))))
        assert rows==6980503
    else:
        candidate=original;rows=6980503
    assets=[ROOT/'frozen-routes.json']+[ROOT/r['panel']/'best-model.pt' for r in routes if r['alpha']>0]
    save(ROOT/'delivery-check.json',dict(passed=True,baseline_files_unchanged=len(preserved),rows=rows,
        speed_values_identical=True,queue_values_identical=True,odme_values_identical=True,
        flow_changes_restricted_to_selected_roads=True,changed_flow_rows=changes,maximum_absolute_flow_change_vph=maximum_delta,
        selected_model_sha256={str(p.relative_to(ROOT)):sha(p) for p in assets},submission=str(candidate.resolve())))


if __name__=='__main__':main()
