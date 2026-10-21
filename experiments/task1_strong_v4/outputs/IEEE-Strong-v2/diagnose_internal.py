"""Development-only flow error slices; no holdout access."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from optimize import ROOT,CACHE

metadata=json.loads((CACHE/'metadata.json').read_text())
rows=[]
for pm in metadata['panels']:
    file=ROOT/pm['panel']/'previous-development.npz'
    if not file.exists():continue
    saved=np.load(file);days=saved['days'];prediction=saved['prediction']
    old=Path(pm['old_cache']);y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
    x=np.load(CACHE/pm['panel']/'features.npy',mmap_mode='r')
    labels=[];pred=[];span=[];hour=[]
    for i,d in enumerate(days):
        m=mask[d];labels.append(y[d,1][m]*600);pred.append(prediction[i,1][m]*600)
        span.append(x[d,14][m]);hour.append(np.broadcast_to(np.arange(288)[:,None]/12,m.shape)[m])
    labels=np.concatenate(labels);pred=np.concatenate(pred);span=np.concatenate(span);hour=np.concatenate(hour)
    cuts=np.quantile(labels,[0,.25,.5,.75,1]);gap_cut=np.median(span)
    groups={f'truth_flow_quartile_{i+1}':(labels>=cuts[i])&(labels<=cuts[i+1] if i==3 else labels<cuts[i+1]) for i in range(4)}
    groups.update(shorter_half_gaps=span<=gap_cut,longer_half_gaps=span>gap_cut,
                  morning_peak=(hour>=6)&(hour<10),evening_peak=(hour>=15)&(hour<19))
    for name,m in groups.items():
        if not m.any():continue
        error=pred[m]-labels[m]
        rows.append(dict(panel=pm['panel'],slice=name,n=int(m.sum()),mean_truth=float(labels[m].mean()),
                         bias=float(error.mean()),rmse=float(np.sqrt(np.mean(error**2)))))
pd.DataFrame(rows).to_csv(ROOT/'internal-flow-error-slices.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.3f}'))
