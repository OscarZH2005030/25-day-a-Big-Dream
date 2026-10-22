"""Recompute the original 51 features from published observations after masking."""
from __future__ import annotations
import numpy as np
import pandas as pd
from task1.transformer_priority.data import reliability_from

COLUMNS=['speed_kmh','flow_vph','occupancy','pct_observed','is_imputed']


def features_from_observations(exp,date,observations,nodes):
    """Exact upstream feature formulas, restricted to requested output nodes.

    Neighbor inputs still use all real nodes, including nodes outside the patch.
    Observations are physical-unit arrays [station,288,5]. No labels enter here.
    """
    nodes=np.asarray(nodes,int)
    if observations.shape!=(exp.n,288,5) or len(np.unique(nodes))!=len(nodes):
        raise ValueError('Observation/patch shape mismatch')
    wd=pd.Timestamp(date).dayofweek
    hist=exp.history[:,nodes,wd]
    ticks=np.broadcast_to(np.arange(288),(len(nodes),288))
    features={'weekday':np.full(ticks.shape,wd),'slot_sin':np.sin(2*np.pi*ticks/288),
              'slot_cos':np.cos(2*np.pi*ticks/288)}
    for col in ['lanes','length_km','free_speed_kmh','capacity_vph']:
        features[col]=np.broadcast_to(exp.static[col].to_numpy()[nodes,None],ticks.shape)
    bases=[]
    for c,col in enumerate(COLUMNS[:2]):
        all_values=observations[:,:,c]
        if c==1:
            all_values=all_values/exp.lanes[:,None]
        a=all_values[nodes]
        valid=np.isfinite(a)
        prev=np.maximum.accumulate(np.where(valid,ticks,-1),axis=1)
        nxt=np.minimum.accumulate(np.where(valid,ticks,288)[:,::-1],axis=1)[:,::-1]
        pv=np.take_along_axis(a,prev.clip(0,287),axis=1)
        nv=np.take_along_axis(a,nxt.clip(0,287),axis=1)
        pv[prev<0]=np.nan;nv[nxt>287]=np.nan
        both=(prev>=0)&(nxt<288)
        fraction=(ticks-prev)/np.maximum(nxt-prev,1)
        initial=np.where(both,pv+(nv-pv)*fraction,hist[c])
        bases.append(initial)
        for suffix,value in [('history',hist[c]),('initial',initial),('prev',pv),('next',nv),
                              ('prev_gap_min',np.where(prev>=0,(ticks-prev)*5,np.nan)),
                              ('next_gap_min',np.where(nxt<288,(nxt-ticks)*5,np.nan)),
                              ('history_fallback',(~both).astype(float))]:
            features[col+'_'+suffix]=value
        for offset in [-6,-3,-1,1,3,6]:
            b=np.full_like(a,np.nan)
            if offset>0: b[:,:-offset]=a[:,offset:]
            else: b[:,-offset:]=a[:,:offset]
            features[f'{col}_offset_{offset*5}min']=b
        for side,groups in exp.neighbors.items():
            neighbor=np.full_like(a,np.nan);available=np.zeros_like(a)
            for j,i in enumerate(nodes):
                ids=groups[i]
                if ids:
                    z=all_values[ids];count=np.isfinite(z).sum(axis=0)
                    neighbor[j]=np.divide(np.nansum(z,axis=0),count,out=np.full(288,np.nan),where=count>0)
                    available[j]=count
            features[f'{col}_{side}']=neighbor;features[f'{col}_{side}_count']=available
    for c,col in enumerate(COLUMNS[2:],start=2):
        a=observations[nodes,:,c]
        if col!='occupancy': features[col]=a
        for offset in [-1,1]:
            b=np.full_like(a,np.nan)
            if offset>0: b[:,:-1]=a[:,1:]
            else: b[:,1:]=a[:,:-1]
            features[f'{col}_offset_{offset*5}min']=b
    features['relative_speed_initial']=bases[0]/exp.static.free_speed_kmh.to_numpy()[nodes,None]
    features['capacity_utilization_initial']=bases[1]*exp.lanes[nodes,None]/exp.static.capacity_vph.to_numpy()[nodes,None]
    raw=np.swapaxes(np.stack(list(features.values()),axis=-1),0,1).astype(np.float32)
    base=np.swapaxes(np.stack([bases[0]/25,bases[1]/600],axis=-1),0,1).astype(np.float32)
    visible=np.swapaxes(np.isfinite(observations[nodes,:,:2]),0,1)
    reliability=reliability_from(raw,list(features),visible)
    return raw,base,visible,reliability,list(features)


def link_groups(exp):
    links=list(dict.fromkeys(exp.links))
    index={link:i for i,link in enumerate(links)}
    node_unit=np.array([index[link] for link in exp.links],int)
    adjacency=[set() for _ in links]
    for groups in exp.neighbors.values():
        for i,neighbors in enumerate(groups):
            for j in neighbors:
                a,b=node_unit[i],node_unit[j]
                if a!=b: adjacency[a].add(b);adjacency[b].add(a)
    return node_unit,adjacency


def additional_mask(observations,node_unit,adjacency,rng,settings,pattern=None):
    eligible=np.isfinite(observations[:,:,:2]).all(-1)&(observations[:,:,3]>=75)
    eligible &= (observations[:,:,:2]>=0).all(-1)
    units=np.zeros((len(adjacency),288),bool)
    np.logical_or.at(units,node_unit,eligible)
    budget=int(np.floor(units.sum()*settings['hidden_fraction']))
    pattern=pattern or settings['patterns'][int(rng.integers(len(settings['patterns'])))]
    if pattern not in settings['patterns']: raise ValueError('Unknown missingness pattern')
    hidden=np.zeros_like(units);fallback=0
    if budget and pattern=='point':
        chosen=rng.choice(np.flatnonzero(units),budget,replace=False);hidden.flat[chosen]=True
    elif budget:
        spatial_seeds=[i for i,neighbors in enumerate(adjacency) if neighbors]
        for _ in range(max(32,4*budget)):
            if hidden.sum()>=budget: break
            if pattern=='spatial' and not spatial_seeds: break
            unit=int(rng.integers(len(adjacency)));start=int(rng.integers(288))
            if pattern=='temporal':
                width=settings['temporal_lengths'][int(rng.integers(len(settings['temporal_lengths'])))]
                selected=[unit]
            else:
                unit=spatial_seeds[int(rng.integers(len(spatial_seeds)))]
                width=settings['spatial_ticks'];selected=[unit];front=[unit]
                while front and len(selected)<settings['spatial_links']:
                    current=front.pop(0)
                    for other in sorted(adjacency[current]):
                        if other not in selected:
                            selected.append(other);front.append(other)
                            if len(selected)>=settings['spatial_links']: break
                if len(selected)==1: continue
            ticks=np.arange(start,min(288,start+width))
            rr,cc=np.nonzero(units[np.ix_(selected,ticks)]&~hidden[np.ix_(selected,ticks)])
            positions=np.asarray(selected)[rr]*288+ticks[cc]
            remaining=budget-int(hidden.sum())
            hidden.flat[positions[:remaining]]=True
        remaining=budget-int(hidden.sum())
        if remaining:
            positions=rng.choice(np.flatnonzero(units&~hidden),remaining,replace=False)
            hidden.flat[positions]=True;fallback=remaining
    mask=hidden[node_unit]
    if not (mask&eligible).any() and budget:
        raise ValueError('Artificial mask has no observed values')
    return mask,eligible,dict(pattern=pattern,eligible_units=int(units.sum()),hidden_units=int(hidden.sum()),point_fill_units=fallback)


def hide_observations(observations,mask):
    result=observations.copy()
    result[mask]=np.nan
    return result
