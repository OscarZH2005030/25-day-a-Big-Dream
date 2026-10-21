"""CPU-only development probe of observed monotone cubic interpolation.

This is an exploratory diagnostic, not part of the frozen model or submission.
It only accesses the original 22 internal development days.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from optimize import ROOT,CACHE,save

metadata=json.loads((CACHE/'metadata.json').read_text());records=[]
for pm in metadata['panels']:
    folder=ROOT/pm['panel'];file=folder/'previous-development.npz'
    if not file.exists():continue
    previous=np.load(file)['prediction'][:,1].astype(np.float64)
    x=np.load(CACHE/pm['panel']/'features.npy',mmap_mode='r')
    old=Path(pm['old_cache']);truth=np.load(old/'y.npy',mmap_mode='r')[196:218,1].astype(np.float64)
    mask=np.load(old/'target.npy',mmap_mode='r')[196:218]
    alternate=previous.copy();ticks=np.arange(288)
    for i,d in enumerate(range(196,218)):
        visible=x[d,6]>0
        for node in range(pm['nodes']):
            valid=visible[:,node]
            if valid.sum()<8:continue
            values=x[d,5,:,node].astype(np.float64)*2000/600
            interp=PchipInterpolator(ticks[valid],values[valid],extrapolate=False)(ticks)
            usable=np.isfinite(interp)&(x[d,14,:,node]<=.5)
            alternate[i,usable,node]=np.maximum(0,interp[usable])
    def errors(p,indices):
        delta=(p-truth)**2;indices=list(indices);m=mask[indices]
        pooled=float(np.sqrt((delta[indices]*m).sum()/m.sum())*600)
        rmse=[]
        for r in ['R1','R2','R3']:
            ids=[i for i in indices if pm['regimes'][i+196]==r]
            if ids:
                mm=mask[ids];rmse.append(np.sqrt((delta[ids]*mm).sum()/mm.sum())*600)
        return dict(pooled=pooled,macro=float(np.mean(rmse)))
    candidates=[]
    for a in [0,.1,.25,.5,.75,1.]:
        value=(1-a)*previous+a*alternate
        candidates.append(dict(alpha=a,all=errors(value,range(22)),early=errors(value,range(11)),late=errors(value,range(11,22))))
    rec=dict(panel=pm['panel'],candidates=candidates)
    records.append(rec)
    print(json.dumps(rec),flush=True)
save(ROOT/'local-flow-probe-development.json',dict(diagnostic_only=True,holdout_opened=False,records=records))
