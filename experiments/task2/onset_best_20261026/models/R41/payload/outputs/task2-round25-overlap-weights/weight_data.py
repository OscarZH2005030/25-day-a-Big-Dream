"""Change label-side window mass only; keep original X/y/IPW and all positives."""
import json,hashlib
from pathlib import Path
import numpy as np
from task2q.cache import atomic_json
def build(p,c,records,work,project):
    folder=work/f'{p}-{c}';folder.mkdir(parents=True,exist_ok=True);source=project/f'work/round10-phase/{p}-{c}';data={k:np.load(source/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','window_positive']};meta=json.loads((source/'complete.json').read_text(encoding='utf8'));done=folder/'complete.json'
    if done.exists():data['w']=np.load(folder/'w.npy',mmap_mode='r');data['w_early']=np.load(folder/'w_early.npy',mmap_mode='r');return data,meta,json.loads(done.read_text(encoding='utf8'))
    lookup={(r['origin'],r['source']):r for r in records if r['panel']==p and r['condition']==c};allcounts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]};earlycounts={b:sum(r['positive']==b and r['early'] for r in meta['windows']) for b in [False,True]};full=np.asarray(data['w']).copy();early=np.zeros_like(full);windows=[]
    for win in meta['windows']:
        r=lookup[(win['origin'],win['source'])];assert r['positive']==win['positive'] and r['early']==win['early'];a,b=win['start'],win['end'];full[a:b]*=r['full_factor']
        if win['early']:early[a:b]=data['w'][a:b]*(allcounts[win['positive']]/earlycounts[win['positive']])*r['early_factor']
        windows.append(dict(origin=win['origin'],source=win['source'],rows=b-a,positive=win['positive'],group=r['group'],group_windows=r['group_windows'],full_factor=r['full_factor'],early_factor=r['early_factor']))
    E=data['early'].astype(bool);P=data['window_positive'].astype(bool);assert data['X'].shape[1]==555;assert (full>0).all() and (early[E]>0).all();assert np.isclose(full.sum(),1,atol=1e-5) and np.isclose(full[P].sum(),.8,atol=1e-5);assert np.isclose(early[E].sum(),1,atol=1e-5) and np.isclose(early[E&P].sum(),.8,atol=1e-5);audit=dict(panel=p,condition=c,rows=len(full),features=555,original_X_y_and_within_window_inverse_inclusion_unchanged=True,all_positive_rows_kept_with_positive_weight=True,positive_window_mass=.8,negative_window_mass=.2,full_mass=float(full.sum()),full_positive_mass=float(full[P].sum()),early_mass=float(early[E].sum()),early_positive_mass=float(early[E&P].sum()),early_normalization_June_September_only=True,source_y_sha256=hashlib.sha256((source/'y.npy').read_bytes()).hexdigest(),source_w_sha256=hashlib.sha256((source/'w.npy').read_bytes()).hexdigest(),weight_exponent=1. if c==0 else .5,windows=windows);np.save(folder/'w.npy',full);np.save(folder/'w_early.npy',early);atomic_json(done,audit);data['w']=full;data['w_early']=early;return data,meta,audit
