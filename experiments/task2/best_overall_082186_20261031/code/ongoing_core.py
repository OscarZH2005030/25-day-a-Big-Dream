"""Causal observed-state, transition and queue-front features for ongoing."""
import numpy as np
import pandas as pd
from onset_interval_core import fine_features

def observed_state(s):
    x=s['x'];cut=s['network']['threshold']/s['network']['free_speed'];good=(x[:,:,3]>=.75)&(x[:,:,7]>.5)&(x[:,:,4]<.5);queue=(x[:,:,0]<=cut[None])&good;n=x.shape[1];age=np.full(n,65.);latest=np.full(n,np.nan)
    for t in range(13):latest=np.where(good[t],x[t,:,0],latest);age=np.where(good[t],(12-t)*5,age)
    known=age<=15;q=(latest<=cut)&known
    return q,known,age,good,queue

def features(s,ramps,counts,basis,context):
    q,known,age,good,queue=observed_state(s);x=s['x'];n=len(q);out={}
    fine,fn=fine_features(s,ramps,counts)
    for i,name in enumerate(fn):out[name]=fine[:,i]
    p,c,pn,cn=context.features(s,ramps,counts)
    for i,name in enumerate(pn):out['ongoing_'+name]=p[:,i]
    for i,name in enumerate(cn):out[name]=c[:,i]
    out['current_observed_queue']=q.astype(float);out['current_state_known']=known.astype(float);out['current_state_age_minutes']=age
    for length in [3,6,12]:
        g=good[-length:];v=queue[-length:];out[f'observed_queue_fraction_{length}']=np.divide(v.sum(0),g.sum(0),out=np.full(n,np.nan),where=g.sum(0)>0);out[f'observed_queue_coverage_{length}']=g.mean(0)
        both=g[1:]&g[:-1];out[f'observed_queue_entries_{length}']=((v[1:]&~v[:-1])&both).sum(0);out[f'observed_queue_exits_{length}']=((~v[1:]&v[:-1])&both).sum(0)
    run=np.zeros(n);active=np.ones(n,bool);started=np.zeros(n,bool)
    for t in range(12,-1,-1):
        begin=~started&good[t];active=np.where(begin,queue[t],np.where(started,active&good[t]&queue[t],active));started|=begin;run+=(active&started)*5
    out['observed_contiguous_queue_run_minutes']=run
    src,dst=s['network']['edges']
    for label,a,b in [('upstream',src,dst),('downstream',dst,src)]:
        degree=np.bincount(b,minlength=n);total=np.zeros(n);valid=np.zeros(n);np.add.at(total,b,q[a]&known[a]);np.add.at(valid,b,known[a]);mean=np.divide(total,valid,out=np.full(n,np.nan),where=valid>0);out[label+'_current_queue_fraction']=mean;out[label+'_queue_state_coverage']=np.divide(valid,degree,out=np.zeros(n),where=degree>0);out[label+'_queue_minus_own']=mean-np.where(known,q,np.nan)
    for k in ['length','lanes','free_speed','capacity','threshold']:
        if k in s['network']:out['static_'+k]=s['network'][k]
    for j in range(8):out[f'position_basis_{j}']=basis[:,j]
    t=pd.Timestamp(s['origin']);angle=2*np.pi*(t.hour*60+t.minute)/1440
    for name,value in [('clock_sin',np.sin(angle)),('clock_cos',np.cos(angle)),('clock_sin2',np.sin(2*angle)),('clock_cos2',np.cos(2*angle)),('weekend',float(t.weekday()>=5))]:out[name]=np.full(n,value)
    xx=np.column_stack(list(out.values())).astype('float32');assert not np.isinf(xx).any();return xx,list(out),q
