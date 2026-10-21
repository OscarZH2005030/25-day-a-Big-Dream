"""Raw-input inference versus the actual 64-channel cache, all ten directions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from inference import features_for_day
from quality_features import quality_features

root=Path(__file__).resolve().parent;workspace=root.parent.parent
cache=workspace/'work/ieee-strong-cache-64'
release=workspace/'work/trafficflowbench-data/kaggle_public'
metadata=json.loads((cache/'metadata.json').read_text())
rows=[]
for pm in metadata['panels']:
    panel=pm['panel'];folder=cache/panel;raw=release/'corridors'/panel
    day=pm['dates'][196]
    source=next(f for f in (raw/'train/mainline_states_masked').rglob('*.parquet')
                if f.stem[-10:].replace('_','-')==day)
    ramp=next(f for f in (raw/'train/ramp_states').rglob('*.parquet')
              if f.stem[-10:].replace('_','-')==day)
    frame=pd.read_parquet(source)
    graph=dict(np.load(folder/'graph.npz'));profile=dict(np.load(folder/'historical-profile.npz'))
    static=pd.read_csv(raw/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
    mapping=pd.read_csv(raw/'network/ramp_attachment_map.csv')
    base=features_for_day(frame,day,pm,static,profile,graph,ramp,mapping)
    extra=quality_features(frame,pm['links'],graph)
    rebuilt=np.concatenate([base,extra],axis=0)
    cached=np.load(folder/'features.npy',mmap_mode='r')[196]
    assert np.array_equal(rebuilt,cached),panel
    target=np.load(Path(pm['old_cache'])/'target.npy',mmap_mode='r')[196]
    assert not extra[1][target].any()
    rows.append(dict(panel=panel,features=64,day=day,bitwise_equal=True,
                     hidden_target_occupancy_exposed=False))
result=dict(passed=True,panels=rows)
(root/'quality-inference-parity.json').write_text(json.dumps(result,indent=2))
print('64-channel raw inference parity passed for all ten directions.')
