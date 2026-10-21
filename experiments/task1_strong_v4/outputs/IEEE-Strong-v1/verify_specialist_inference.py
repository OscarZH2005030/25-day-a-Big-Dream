"""Check final raw-input routing against saved benchmark predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from inference import features_for_day
from quality_features import quality_features
from hybrid_model import make_model

root=Path(__file__).resolve().parent;workspace=root.parent.parent
assert (root/'specialists-complete.json').exists(),'Wait for all GPU training to finish.'
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
manifest=json.loads((root/'final-model.json').read_text())
assets=root/'inference-assets-64'
asset_manifest=json.loads((assets/'manifest.json').read_text())
panels={p['panel']:p for p in asset_manifest['panels']}
cache=workspace/'work/ieee-strong-cache-64'
release=workspace/'work/trafficflowbench-data/kaggle_public'
records=[];model=None
for route in manifest['routes']:
    panel=route['panel'];pm=panels[panel];folder=release/'corridors'/panel
    day='2031-01-05'
    raw_file=next(f for f in (folder/'train/mainline_states_masked').rglob('*.parquet')
                  if f.stem[-10:].replace('_','-')==day)
    ramp_file=next(f for f in (folder/'train/ramp_states').rglob('*.parquet')
                   if f.stem[-10:].replace('_','-')==day)
    raw=pd.read_parquet(raw_file)
    static=pd.read_csv(folder/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
    mapping=pd.read_csv(folder/'network/ramp_attachment_map.csv')
    profile=dict(np.load(assets/panel/'historical-profile.npz'))
    feature_graph=dict(np.load(assets/panel/'graph.npz'))
    x=features_for_day(raw,day,pm,static,profile,feature_graph,ramp_file,mapping)
    x=np.concatenate([x,quality_features(raw,pm['links'],feature_graph)],axis=0)
    assert np.array_equal(x,np.load(cache/panel/'features.npy',mmap_mode='r')[218])
    checkpoint=torch.load(root/route['checkpoint'],map_location='cuda',weights_only=True)
    if model is None:model=make_model(checkpoint['config'],64).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    graph={k:torch.from_numpy(v).cuda() for k,v in np.load(root/route['model_graph']).items()}
    graph['node_id']=torch.arange(pm['offset'],pm['offset']+pm['nodes'],device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        pred=model(torch.from_numpy(x[None].astype(np.float32)).cuda(),graph)[0].cpu().numpy()
    with np.load((root/route['checkpoint']).parent/(panel+'-holdout.npz')) as saved:
        assert saved['days'][0]==218
        difference=float(np.abs(pred-saved['prediction'][0]).max())
    assert difference<=2e-4,(panel,difference)
    records.append(dict(panel=panel,raw_features_bitwise_equal=True,
        maximum_normalized_prediction_difference=difference,correct_model_graph=True))
(root/'specialist-inference-parity.json').write_text(json.dumps(dict(passed=True,
    prediction_tolerance=2e-4,checks=records),indent=2))
print('All ten routed raw-input predictions match their saved benchmark outputs within tolerance.')
