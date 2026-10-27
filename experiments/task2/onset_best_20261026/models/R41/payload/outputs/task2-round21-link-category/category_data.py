"""Add a public static link category; replay R10 cells/labels/IPW exactly."""
import os,json,hashlib
from pathlib import Path
import numpy as np
from coverage_data import matrix as original_matrix
from special_features import latest_state
from run_round2 import sample,PANELS
from task2q.cache import atomic_json
SEED=20261027
def features(s):
    n=s['x'].shape[1];return np.concatenate([original_matrix(s),np.tile(np.arange(n,dtype='float32'),6)[:,None]],axis=1)
def build(entries,p,work,project,progress):
    folder=work/p;folder.mkdir(parents=True,exist_ok=True);base=project/f'work/round10-phase/{p}-0';data={k:np.load(base/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','window_positive']};meta=json.loads((base/'complete.json').read_text(encoding='utf8'));done=folder/'complete.json'
    if done.exists():data['X']=np.load(folder/'X.npy',mmap_mode='r');return data,meta,json.loads(done.read_text(encoding='utf8'))
    dest=folder/'building.npy';X=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(data['y']),556));lookup={(e['origin'],e['source']):e for e in entries if e['panel']==p and e['condition']==0};rngs={source:np.random.default_rng(SEED+offset+PANELS.index(p)*2) for source,offset in [('original',0),('added_phase',2000)]};cursor=0
    for i,win in enumerate(meta['windows']):
        e=lookup[(win['origin'],win['source'])];s=sample(e);n=s['x'].shape[1];m=s['m'].ravel();y=s['y'].ravel();last,_=latest_state(s);mandatory=m&(y>0);keep=np.flatnonzero(mandatory);negative=np.flatnonzero(m&~mandatory);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.2),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));rng=rngs[win['source']];aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);start,end=win['start'],win['end'];assert start==cursor and len(ids)==end-start;assert np.array_equal(y[ids],data['y'][start:end]);assert np.array_equal(original_matrix(s)[ids],data['X'][start:end]);X[start:end,:555]=data['X'][start:end];X[start:end,555]=ids%n;cursor=end
        if (i+1)%128==0:progress('replaying_link_categories',panel=p,windows=i+1,total=len(meta['windows']),rows=cursor)
    assert cursor==len(data['y']) and np.isfinite(X).all();X.flush();del X;os.replace(dest,folder/'X.npy');audit=dict(panel=p,condition=0,rows=cursor,features=556,original555_and_labels_and_IPW_identical=True,categorical_column=555,source_y_sha256=hashlib.sha256((base/'y.npy').read_bytes()).hexdigest(),source_w_sha256=hashlib.sha256((base/'w.npy').read_bytes()).hexdigest(),identity='index in frozen public network.link_ids; repeated horizon-major, never a target-derived prior');atomic_json(done,audit);data['X']=np.load(folder/'X.npy',mmap_mode='r');return data,meta,audit
