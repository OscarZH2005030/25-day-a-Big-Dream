"""Reconstruct only inside the supplied <=T history; retain observation masks."""
import numpy as np

def repair(values, observed, default, linear=True):
    values=np.asarray(values,dtype=np.float32); observed=np.asarray(observed,dtype=bool)
    t,n=values.shape; out=np.full((t,n),default,dtype=np.float32)
    for j in range(n):
        known=np.flatnonzero(observed[:,j])
        if not len(known): continue
        if linear:
            out[:,j]=np.interp(np.arange(t),known,values[known,j])
        else:
            for i in range(t):
                past=known[known<=i]
                if len(past): out[i,j]=values[past[-1],j]
    return out

def repaired(s):
    x=s['x']; values=[]; masks=[]
    for channel,missing,default in [(0,4,1.),(1,5,0.),(2,6,0.)]:
        observed=x[:,:,missing]<.5
        values.append(repair(x[:,:,channel],observed,default));masks.append(observed)
    return values,masks

def extras(s):
    vals,masks=repaired(s); n=s['x'].shape[1]; columns=[]
    for v,m in zip(vals,masks):
        columns.extend([v.T, m.T.astype('float32')])
        for k in [3,6,13]:
            a=v[-k:]; columns.append(np.stack([a.mean(0),a.std(0),a.min(0),a.max(0),a[-1]-a[0],m[-k:].mean(0)],axis=1))
    speed=vals[0];flow=vals[1];cut=s['network']['threshold']/s['network']['free_speed']
    columns.append(np.stack([(speed<=cut).mean(0),(speed[-6:]<=cut).mean(0),speed[-1]-cut,flow[-1]-flow[-3]],axis=1))
    base=np.concatenate(columns,axis=1).astype('float32');assert base.shape==(n,136)
    return np.tile(base,(6,1))
