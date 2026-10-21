"""Statistical proximity fallback for nodes isolated in the actual topology.

The physical network is not modified. Weights use dimensionless relative
milepost gaps, so no assumption about milepost units is required.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent;workspace=root.parent.parent
cache=workspace/'work/ieee-strong-cache-64'
release=workspace/'work/trafficflowbench-data/kaggle_public'
meta=json.loads((cache/'metadata.json').read_text());records=[]
for pm in meta['panels']:
    name=pm['panel'];original=dict(np.load(cache/name/'graph.npz'))
    graph={k:v.copy() for k,v in original.items()}
    f=next((release/'corridors'/name/'train/mainline_states_masked').rglob('*.parquet'))
    milepost=pd.read_parquet(f,columns=['link_id','milepost']).groupby('link_id').milepost.first().reindex(pm['links']).to_numpy(float)
    isolated=(original['up_weight'].sum(-1)+original['down_weight'].sum(-1))==0
    delta=milepost[:,None]-milepost[original['up_index']]
    usable=(original['up_weight']>0)&np.isfinite(delta)&(np.abs(delta)>1e-8)
    ascending=float(np.median(delta[usable]))>0 if usable.any() else name.endswith(('_N','_E'))
    order=np.argsort(milepost)
    order=order if ascending else order[::-1]
    gaps=np.abs(np.diff(milepost[order]));positive=gaps[np.isfinite(gaps)&(gaps>1e-8)]
    scale=float(np.median(positive)) if len(positive) else 1.
    added=0
    for pos,node in enumerate(order):
        if not isolated[node] or not np.isfinite(milepost[node]):continue
        for offset,prefix in [(-1,'up'),(1,'down')]:
            neighbor_pos=pos+offset
            if not 0<=neighbor_pos<len(order):continue
            adjacent=order[neighbor_pos];gap=abs(milepost[node]-milepost[adjacent])
            if not np.isfinite(gap) or gap>3*scale:continue
            graph[prefix+'_index'][node,0]=adjacent
            graph[prefix+'_weight'][node,0]=.5*np.exp(-gap/scale)
            added+=1
    dest=root/'proximity-assets'/name;dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'graph.npz',**graph)
    remaining=(graph['up_weight'].sum(-1)+graph['down_weight'].sum(-1))==0
    assert all(np.array_equal(graph[k][~isolated],original[k][~isolated]) for k in original)
    records.append(dict(panel=name,nodes=pm['nodes'],initially_isolated=int(isolated.sum()),
        remaining_isolated=int(remaining.sum()),added_statistical_relations=added,
        median_positive_milepost_gap=scale,ascending=ascending,
        physical_connected_rows_unchanged=True))
(root/'proximity-assets/manifest.json').write_text(json.dumps(dict(
    formula='0.5*exp(-gap/median_positive_gap), only gaps <=3*median_positive_gap; only globally isolated receivers',
    selection='Compare original and supplemented model graph on internal development only',
    features='Cached input features always retain the original physical graph',panels=records),indent=2))
print(json.dumps(records,indent=2))
