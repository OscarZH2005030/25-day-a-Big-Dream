"""Reuse original R10 matrices and sampling; only objective changes."""
import json
from pathlib import Path
import numpy as np
from task2q.cache import atomic_json
def build(p,c,unused,work,project):
    assert c==0
    folder=project/f'work/round10-phase/{p}-{c}'
    data={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','window_positive']}
    meta=json.loads((folder/'complete.json').read_text(encoding='utf8'))
    total={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]}
    early={b:sum(r['positive']==b and r['early'] for r in meta['windows']) for b in [False,True]}
    ew=np.zeros_like(data['w']);E=data['early'].astype(bool);P=data['window_positive'].astype(bool)
    for b in [False,True]:
        ii=E&(P==b);ew[ii]=data['w'][ii]*total[b]/early[b]
    data['w_early']=ew
    assert np.isclose(ew.sum(),1,atol=1e-5) and np.isclose(ew[P].sum(),.8,atol=1e-5)
    audit=dict(panel=p,condition=c,rows=len(data['y']),features=555,original_X_y_w_and_sampling_unchanged=True,early_positive_windows=early[True],early_mass=float(ew.sum()),early_positive_mass=float(ew[P].sum()),early_normalization_June_September_only=True,windows=meta['windows'])
    return data,meta,audit
def selected_windows(meta,early=False):
    result=[];cursor=0
    for win in meta['windows']:
        if early and not win['early']:continue
        n=win['end']-win['start'];result.append(dict(start=cursor,end=cursor+n,positive=win['positive']));cursor+=n
    return result
