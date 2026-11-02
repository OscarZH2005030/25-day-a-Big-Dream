import sys,time
sys.path.insert(0,r'C:\Users\ROG\Desktop\IEEE Big data 2026\task2_features\experiments\task2\r127_reproduction_20261027');sys.path.insert(0,r'C:\Users\ROG\Desktop\IEEE Big data 2026\task2_features\f12')
from repro_common import *
from pathlib import Path
import numpy as np,copy
from f12_features import link_features
OUT=Path(r'C:\Users\ROG\tfb-r127-work\f12_extra');OUT.mkdir(parents=True,exist_ok=True)
model,sample,r2,audit=setup(Path(r'C:\Users\ROG\tfb-r127-assets\ConditionalPosition-R127'),Path(r'C:\Users\ROG\Desktop\IEEE Big data 2026\25-day-a-Big-Dream\data\kaggle_public'),Path(r'C:\Users\ROG\tfb-r127-work\cache'),True,HERE/'indices/training-array-hashes.json',[OUT,Path(r'C:\Users\ROG\tfb-r127-work')])
protocol=read(HERE/'training/raw-rebuild-recipe.json');checked=0
def causal_check(s):
    t=copy.copy(s);rng=np.random.default_rng(1)
    for k in ['y','m','label_mask','soft_y','speed_y']:t[k]=rng.random(s[k].shape).astype(s[k].dtype) if s[k].dtype!=bool else rng.random(s[k].shape)>.5
    t['has_future_queue']=not s['has_future_queue']
    a=link_features(s['x'],s['network']);b=link_features(t['x'],t['network'])
    assert all(np.array_equal(u,v,equal_nan=True) for u,v in zip(a,b)),'feature depends on future fields'
for p,path in protocol['position_recipes'].items():
    recipe=read(HERE/path);n_rows=recipe['windows'][-1]['end'];F=np.full((n_rows,14),np.nan,np.float32);t0=time.time()
    for i,row in enumerate(recipe['windows']):
        s=sample(row['entry']);ids=np.asarray(row['selected_cells'],dtype=int);n=s['x'].shape[1];F1,F2=link_features(s['x'],s['network']);F[row['start']:row['end']]=np.column_stack([F1,F2])[ids%n]
        if i%100==0:causal_check(s);checked+=1
        if (i+1)%100==0:print(p,i+1,len(recipe['windows']),round(time.time()-t0),flush=True)
    np.save(OUT/(p+'_F.npy'),F)
rows=read(HERE/protocol['conditional_rows']);groups={}
for i,row in enumerate(rows):groups.setdefault((row['panel'],row['origin']),[]).append((i,row))
F=np.full((len(rows),14),np.nan,np.float32)
for i,rr in enumerate(groups.values()):
    s=sample(rr[0][1]);ids=np.array([r['target_link_index'] for j,r in rr]);ix=np.array([j for j,r in rr]);F1,F2=link_features(s['x'],s['network']);F[ix]=np.column_stack([F1,F2])[ids]
    if i%500==0:causal_check(s);checked+=1;print('cond',i,len(groups),flush=True)
np.save(OUT/'conditional_F.npy',F);print('DONE causality_checks_passed',checked)
