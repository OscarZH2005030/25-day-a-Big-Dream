"""Fixed-threshold event proposal, conditional temporal decode and verification."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/task2-gate'))
sys.path.insert(0,str(ROOT/'work/task2-rescue-v2'))
from gate import r,gate_features
from context import Context
from experts import augmented_matrix

def first_allowed(y,m):
    y=np.asarray(y,dtype=bool);m=np.asarray(m,dtype=bool);n=len(y)
    positive=y&m;has=positive.any(1);j=np.argmax(positive,axis=1);a=np.zeros((n,7),bool);a[:,0]=~has
    for k in range(6):a[:,k+1]=(~m[:,k]|positive[:,k])&(~has|(k<=j))
    assert a.any(1).all();return a

def softmax(raw):
    v=np.exp(raw-raw.max(1,keepdims=True));return v/v.sum(1,keepdims=True)

def partial_loss(raw,a,w=None):
    p=softmax(raw);post=softmax(np.where(a,raw,-np.inf));g=p-post;h=np.maximum(2*p*(1-p),1e-4);loss=-np.log(np.maximum((p*a).sum(1),1e-300))
    if w is not None:return float(np.average(loss,weights=w)),g*w[:,None],h*w[:,None]
    return float(loss.mean()),g,h

def physical(s):
    x=s['x'];n=x.shape[1];cut=s['network']['threshold']/s['network']['free_speed']
    def carry(channel,missing,end,default):
        v=np.full(n,default,dtype=float);known=np.zeros(n,bool)
        for t in range(end+1):
            ok=(x[t,:,missing]<.5)&(x[t,:,3]>=.75)&(x[t,:,7]>.5);v[ok]=x[t,ok,channel];known|=ok
        return v,known
    v,kv=carry(0,4,12,1.);v15,ko=carry(0,4,9,1.);f,kf=carry(1,5,12,0.);f15,_=carry(1,5,9,0.);o,ko2=carry(2,6,12,0.);o15,_=carry(2,6,9,0.)
    dv=v-v15;df=f-f15;do=o-o15;down=np.zeros(n);up=np.zeros(n);dd=np.zeros(n);ud=np.zeros(n);dn=np.zeros(n);un=np.zeros(n)
    for source,target in s['network']['edges'].T:
        if kv[target]:down[source]=max(down[source],max(0.,cut[target]-v[target]));dd[source]=max(dd[source],max(0.,-dv[target]));dn[source]+=1
        if kv[source]:up[target]=max(up[target],max(0.,cut[source]-v[source]));ud[target]=max(ud[target],max(0.,-dv[source]));un[target]+=1
    valid=(x[:,:,4]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);near=valid&(x[:,:,0]<=cut+.15)
    run=np.zeros(n);longest=np.zeros(n)
    for t in range(13):run=np.where(near[t],run+1,0);longest=np.maximum(longest,run)
    values=np.column_stack([dv,df,do,np.clip(f/np.maximum(v,.1),0,10),np.maximum(-dv,0)*np.maximum(do,0),down,up,dd,ud,dn/4,un/4,longest/13,valid.mean(0),kv&kf&ko2,near.mean(0),f])
    names=['speed_change15','flow_change15','occupancy_change15','density_to_critical_proxy','deceleration_occupancy_agreement','downstream_queue_pressure','upstream_queue_pressure','downstream_deceleration','upstream_deceleration','known_downstream_neighbors','known_upstream_neighbors','persistent_near_queue','history_speed_coverage','joint_channel_evidence','near_queue_fraction','current_demand_ratio']
    return values.astype('float32'),names

def verifier_features(s,context,eprob,q,b,regime):
    n=s['x'].shape[1];zero=np.zeros_like(q);g=gate_features(s,zero,q);keep=[i for i in range(44) if i not in [26,27,28,29,30,32,34]];extra,future=context.extra(s,regime);ph,pnames=physical(s)
    link=np.column_stack([g[:,keep],extra,ph]);h=np.arange(6);event=1-eprob[:,0];first=eprob[:,1:];cdf=np.cumsum(first,axis=1);expect=(first*np.arange(6)).sum(1)/np.maximum(event,1e-8)/5
    seed=b>=.5;distance=np.full((6,n),1.)
    for i in range(n):
        ids=np.flatnonzero(seed[:,i])
        if len(ids):distance[:,i]=np.abs(h[:,None]-ids[None,:]).min(1)/5
    prev=np.vstack([q[:1],q[:-1]]);nxt=np.vstack([q[1:],q[-1:]])
    temporal=np.stack([q,prev,nxt,np.tile(q.max(0),(6,1)),np.tile(q.mean(0),(6,1)),np.tile((q>=.5).mean(0),(6,1)),np.tile(event,(6,1)),first.T,cdf.T,np.tile(expect,(6,1)),np.tile(seed.any(0),(6,1)),np.tile(seed.mean(0),(6,1)),distance],axis=2).transpose(1,0,2)
    matrix=np.concatenate([np.tile(link[:,None,:],(1,6,1)),future,np.tile(np.eye(6)[None],(n,1,1)),temporal],axis=2).astype('float32')
    assert np.isfinite(matrix).all()
    return matrix,dict(physical_start=37+40,physical_names=pnames,columns=matrix.shape[2])
