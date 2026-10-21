"""Collapse public paths through unsensed links into observed-node relations.

Only receivers isolated in the observed physical subgraph are supplemented.
These are path relations for the predictor, not new direct LWR physics edges.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent;workspace=root.parent.parent
cache=workspace/'work/ieee-strong-cache-64'
release=workspace/'work/trafficflowbench-data/kaggle_public'
meta=json.loads((cache/'metadata.json').read_text());records=[]
for pm in meta['panels']:
    name=pm['panel'];index={v:i for i,v in enumerate(pm['links'])}
    original=dict(np.load(cache/name/'graph.npz'));graph={k:v.copy() for k,v in original.items()}
    isolated=(original['up_weight'].sum(-1)+original['down_weight'].sum(-1))==0
    choices={'up':defaultdict(dict),'down':defaultdict(dict)}
    for sequence in pd.read_csv(release/'corridors'/name/'network/path_set.csv',usecols=['link_seq']).link_seq:
        previous=None
        for step,link in enumerate(str(sequence).split(';')):
            if link not in index:continue
            node=index[link]
            if previous is not None:
                old_step,old_node=previous;gap=step-old_step
                if old_node!=node and gap<=8:
                    for receiver,sender,prefix in [(node,old_node,'up'),(old_node,node,'down')]:
                        if isolated[receiver]:
                            prior=choices[prefix][receiver].get(sender,(999,0))
                            choices[prefix][receiver][sender]=(min(prior[0],gap),prior[1]+1)
            previous=(step,node)
    added=0
    for prefix in ['up','down']:
        for receiver,candidates in choices[prefix].items():
            # Nearest along actual path; path support breaks equal-hop ties.
            sender,(gap,support)=min(candidates.items(),key=lambda v:(v[1][0],-v[1][1],v[0]))
            graph[prefix+'_index'][receiver,0]=sender
            graph[prefix+'_weight'][receiver,0]=.5*np.exp(-(gap-1)/4)
            added+=1
    assert all(np.array_equal(graph[k][~isolated],original[k][~isolated]) for k in original)
    dest=root/'path-assets'/name;dest.mkdir(parents=True,exist_ok=True)
    np.savez(dest/'graph.npz',**graph)
    remaining=(graph['up_weight'].sum(-1)+graph['down_weight'].sum(-1))==0
    records.append(dict(panel=name,initially_isolated=int(isolated.sum()),
        remaining_isolated=int(remaining.sum()),added_path_relations=added,
        connected_physical_rows_unchanged=True))
(root/'path-assets/manifest.json').write_text(json.dumps(dict(
    source='Public path_set.csv link_seq only; no dynamic observations or labels',
    relation='Consecutive observed nodes on a public path, separated by <=8 path links; minimum gap then maximum path support',
    weight='0.5*exp(-(gap-1)/4)',panels=records),indent=2))
print(json.dumps(records,indent=2))
