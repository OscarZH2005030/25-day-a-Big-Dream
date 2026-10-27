"""New context along public path reachability; original immediate graph stays intact."""
import os,json,hashlib
from pathlib import Path
import numpy as np
from coverage_data import matrix as original_matrix
from special_features import latest_state
from run_round2 import sample,PANELS
from task2q.cache import atomic_json
SEED=20261027
def context(s,graph):
    x=s['x'];n=x.shape[1];assert [str(l) for l in s['network']['link_ids']]==graph['link_ids'];edges=graph['edges'];blocks=[];cap=s['network']['capacity'];cut=s['network']['threshold']/s['network']['free_speed'];quality=(x[:,:,3]>=.75)&(x[:,:,7]>.5);valid=[quality&(x[:,:,missing]<.5) for missing in [4,5,6]];raw=[x[:,:,0],x[:,:,1]*cap[None],x[:,:,2]]
    for reverse in [False,True]:
        src=np.array([e['target' if reverse else 'source'] for e in edges],dtype=int);dst=np.array([e['source' if reverse else 'target'] for e in edges],dtype=int);degree=np.bincount(dst,minlength=n);gap=np.array([e['min_gap'] for e in edges],dtype=float);witness=np.array([e['witness_path_count'] for e in edges],dtype=float);total_gap=np.zeros(n);min_gap=np.full(n,np.inf);max_gap=np.zeros(n);witness_total=np.zeros(n);min_cap=np.full(n,np.inf);mean_cap=np.zeros(n)
        if len(src):
            np.add.at(total_gap,dst,gap);np.minimum.at(min_gap,dst,gap);np.maximum.at(max_gap,dst,gap);np.add.at(witness_total,dst,witness);np.minimum.at(min_cap,dst,cap[src]);np.add.at(mean_cap,dst,cap[src])
        profile=np.zeros((13,n,9),dtype='float32');counts=[]
        for ch in range(3):
            value=raw[ch];known=valid[ch];sums=np.zeros((13,n));count=np.zeros((13,n))
            if len(src):np.add.at(sums,(np.arange(13)[:,None],dst[None]),np.where(known[:,src],value[:,src],0));np.add.at(count,(np.arange(13)[:,None],dst[None]),known[:,src])
            mean=np.divide(sums,count,out=np.full_like(sums,1. if ch==0 else 0.),where=count>0)
            if ch==1:mean/=cap[None]
            profile[:,:,[0,2,4][ch]]=mean;profile[:,:,6+ch]=np.divide(count,degree[None],out=np.zeros_like(count),where=degree[None]>0);counts.append(count)
        minimum=np.full((13,n),np.inf);maximum=np.zeros((13,n));pressure=np.zeros((13,n))
        if len(src):
            index=(np.arange(13)[:,None],dst[None]);np.minimum.at(minimum,index,np.where(valid[0][:,src],raw[0][:,src],np.inf));np.maximum.at(maximum,index,np.where(valid[1][:,src],raw[1][:,src],0));np.maximum.at(pressure,index,np.where(valid[0][:,src],np.maximum(cut[src][None]-raw[0][:,src],0),0))
        profile[:,:,1]=np.where(np.isfinite(minimum),minimum,1.);profile[:,:,3]=maximum/cap[None];profile[:,:,5]=pressure;static=np.column_stack([degree/4,np.divide(total_gap,degree,out=np.zeros(n),where=degree>0)/8,np.where(np.isfinite(min_gap),min_gap,0)/8,max_gap/8,np.log1p(witness_total),np.where(np.isfinite(min_cap),min_cap,0)/cap,np.divide(mean_cap,degree,out=np.zeros(n),where=degree>0)/cap]);trend=profile[-1,:,[0,2,3,5]].T-profile[-4,:,[0,2,3,5]].T;block=np.concatenate([profile.transpose(1,0,2).reshape(n,-1),static,trend],axis=1);assert block.shape==(n,128);blocks.append(block)
    result=np.tile(np.concatenate(blocks,axis=1),(6,1)).astype('float32');assert result.shape==(6*n,256) and np.isfinite(result).all();return result
def features(s,graph):return np.concatenate([original_matrix(s),context(s,graph)],axis=1)
def build(entries,p,c,graph,work,project,progress):
    folder=work/f'{p}-{c}';folder.mkdir(parents=True,exist_ok=True);base=project/f'work/round10-phase/{p}-{c}';data={k:np.load(base/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','window_positive']};meta=json.loads((base/'complete.json').read_text(encoding='utf8'));done=folder/'complete.json'
    if done.exists():data['X']=np.load(folder/'X.npy',mmap_mode='r');return data,meta,json.loads(done.read_text(encoding='utf8'))
    dest=folder/'building.npy';X=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(data['y']),811));lookup={(e['origin'],e['source']):e for e in entries if e['panel']==p and e['condition']==c};rngs={source:np.random.default_rng(SEED+offset+PANELS.index(p)*2+c) for source,offset in [('original',0),('added_phase',2000)]};cursor=0
    for i,win in enumerate(meta['windows']):
        e=lookup[(win['origin'],win['source'])];s=sample(e);m=s['m'].ravel();y=s['y'].ravel();last,queued=latest_state(s);flip=y.astype(bool)!=np.tile(queued,6);mandatory=(m&(y>0))|(m&flip if c==1 else False);keep=np.flatnonzero(mandatory);negative=np.flatnonzero(m&~mandatory);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.2),6);a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a));rng=rngs[win['source']];aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int);ids=np.concatenate([keep,aa,bb]);start,end=win['start'],win['end'];assert start==cursor and len(ids)==end-start;assert np.array_equal(y[ids],data['y'][start:end]);assert np.array_equal(original_matrix(s)[ids],data['X'][start:end]);X[start:end,:555]=data['X'][start:end];X[start:end,555:]=context(s,graph)[ids];cursor=end
        if (i+1)%128==0:progress('replaying_path_rows',panel=p,condition=c,windows=i+1,total=len(meta['windows']),rows=cursor)
    assert cursor==len(data['y']) and np.isfinite(X).all();X.flush();del X;os.replace(dest,folder/'X.npy');audit=dict(panel=p,condition=c,rows=cursor,features=811,original555_labels_and_IPW_identical=True,path_features=256,source_y_sha256=hashlib.sha256((base/'y.npy').read_bytes()).hexdigest(),source_w_sha256=hashlib.sha256((base/'w.npy').read_bytes()).hexdigest(),semantics='mean/max known endpoint observations along witnessed directed paths, not hidden-state reconstruction, exact adjacency, or closed flow conservation');atomic_json(done,audit);data['X']=np.load(folder/'X.npy',mmap_mode='r');return data,meta,audit
