"""Causal directed sequence evidence, including explicit missing coverage.

The graph determines direction only; edge indices are not physical distances.
No labels, future measurement arrays or real-world calendar data are read.
"""
import numpy as np

def features(s):
    x=np.asarray(s['x'],float);T,n,_=x.shape;assert T==13
    cut=s['network']['threshold']/s['network']['free_speed'];src,dst=np.asarray(s['network']['edges'],int)
    good=(x[:,:,3]>=.75)&(x[:,:,7]>.5);v=good&(x[:,:,4]<.5)&np.isfinite(x[:,:,0]);o=good&(x[:,:,6]<.5)&np.isfinite(x[:,:,2])
    gap=x[:,:,0]-cut;queued=(gap<=0)&v;near=np.maximum(-gap,0)*v
    cols=[];names=[]
    def add(name,a):names.append(name);cols.append(np.asarray(a))
    def avg(a,m):return np.divide(np.where(m,a,0).sum(0),m.sum(0),out=np.zeros(a.shape[1]),where=m.sum(0)>0)
    latest=np.ones(n);age=np.ones(n)*13
    for t in range(T):latest[v[t]]=gap[t,v[t]];age[v[t]]=T-1-t
    add('local_latest_gap',latest);add('local_observation_age',age/13)
    for span in [3,6,12]:
        add(f'local_coverage_{span}',v[-span:].mean(0));add(f'local_gap_{span}',avg(gap[-span:],v[-span:]));add(f'local_queue_share_{span}',queued[-span:].mean(0))
    # Downstream lead -> local response is backward queue propagation. The
    # opposite direction is also retained as a competing explanation.
    for direction,a,b in [('downstream',src,dst),('upstream',dst,src)]:
        degree=np.bincount(a,minlength=n);den=np.maximum(degree,1)
        aggregate=[]
        for field in [v.astype(float),queued.astype(float),near]:
            out=np.zeros((T,n))
            for t in range(T):np.add.at(out[t],a,field[t,b])
            aggregate.append(out/den)
        coverage,qs,pressure=aggregate
        add(direction+'_no_neighbors',degree==0)
        for span in [1,3,6,12]:
            add(f'{direction}_coverage_{span}',coverage[-span:].mean(0));add(f'{direction}_queue_share_{span}',qs[-span:].mean(0));add(f'{direction}_queue_pressure_{span}',pressure[-span:].mean(0))
        add(direction+'_queue_growth_15',qs[-1]-qs[-4]);add(direction+'_pressure_growth_15',pressure[-1]-pressure[-4])
        # Correlations are computed within the SAME edge and matched timestamp
        # pairs. Maxima from unrelated edges/timestamps are never multiplied.
        dec=x[:-1,:,0]-x[1:,:,0];dm=v[:-1]&v[1:];rise=x[1:,:,2]-x[:-1,:,2];om=o[:-1]&o[1:]
        for lag in [0,1,2,3]:
            totals=np.zeros((n,5));count=np.zeros(n)
            for local,neighbor in zip(a,b):
                la=np.arange(lag,T-1);nb=la-lag;mask=dm[la,local]&dm[nb,neighbor]
                if not mask.any():continue
                d1=dec[la,local][mask];d2=dec[nb,neighbor][mask]
                co=float(d1@d2)/max(float(np.linalg.norm(d1)*np.linalg.norm(d2)),1e-6)
                both=mask&om[la,local];joint=np.maximum(dec[nb,neighbor],0)*np.maximum(rise[la,local],0)
                totals[local]+=[co,np.mean(np.maximum(d2,0)),np.mean(np.maximum(d1,0)*np.maximum(d2,0)),np.mean(joint[both]) if both.any() else 0,mask.mean()];count[local]+=1
            for j,label in enumerate(['deceleration_cosine','neighbor_deceleration','matched_deceleration','neighbor_deceleration_local_occupancy','paired_coverage']):
                add(f'{direction}_leads_{lag*5}min_{label}',totals[:,j]/np.maximum(count,1))
        # Two-hop queue support adds extent, without assuming metres per edge.
        out=np.zeros(n)
        for local,neighbor in zip(a,b):out[local]+=qs[-1,neighbor]
        add(direction+'_two_hop_queue_share',out/den)
    out=np.column_stack(cols).astype('float32');assert np.isfinite(out).all()
    return out,names

def horizon_features(raw,h):
    # Existing physical queue line is unchanged; horizon interactions are
    # learned features, never decision cutoffs or inferred physical travel time.
    t=(h+1)/6
    return np.r_[raw,raw*t,raw*(1-t)].astype('float32')
