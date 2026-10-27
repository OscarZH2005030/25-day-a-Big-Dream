import numpy as np
import torch

def graph_tensors(graph,nodes):
    nodes=np.asarray(nodes,dtype=int)
    if len(np.unique(nodes))!=len(nodes):
        raise ValueError('Duplicate nodes in patch')
    sub=graph['graph'][np.ix_(nodes,nodes)]
    width=int(sub.sum(1).max())
    neighbors=np.zeros((len(nodes),width),dtype=np.int64)
    valid=np.zeros_like(neighbors,dtype=bool);direction=np.zeros_like(neighbors)
    for i,row in enumerate(sub):
        ids=np.flatnonzero(row);neighbors[i,:len(ids)]=ids;valid[i,:len(ids)]=True
        direction[i,:len(ids)]=graph['direction'][nodes[i],nodes[ids]]+1
    return dict(neighbors=torch.from_numpy(neighbors),neighbor_valid=torch.from_numpy(valid),direction=torch.from_numpy(direction))

def sampling_spec(rng,n,order,dates,window,patch,batch,balanced):
    width=min(n,patch)
    start=int(rng.integers(n if balanced else n-width+1))
    positions=(start+np.arange(width))%n if balanced else np.arange(start,start+width)
    nodes=order[positions]
    samples=[]
    for _ in range(batch):
        date=dates[int(rng.integers(len(dates)))]
        tick=int(rng.integers(-(window-1),288)) if balanced else int(rng.integers(289-window))
        samples.append((date,tick))
    return nodes,samples

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

def choose_supervision(original,artificial,labels,extra_values,rng,fraction):
    original=np.asarray(original,bool);artificial=np.asarray(artificial,bool)&~original
    count=int(original.sum())
    extra_count=min(int(np.floor(count*fraction)),int(artificial.sum()))
    target=original.copy();label=labels.copy()
    if extra_count:
        removed=rng.choice(np.flatnonzero(original),extra_count,replace=False)
        added=rng.choice(np.flatnonzero(artificial),extra_count,replace=False)
        target.flat[removed]=False;target.flat[added]=True
        label.reshape(-1,2)[added]=extra_values.reshape(-1,2)[added]
    if target.sum()!=count or (target&original).sum()<count*.8-1e-8:
        raise ValueError('Dynamic supervision budget changed')
    return target,label