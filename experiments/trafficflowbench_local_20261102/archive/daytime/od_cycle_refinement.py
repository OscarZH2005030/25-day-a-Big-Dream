"""Feasible L1-target cycle updates, preserving OD margins and prior L1 distance."""
import numpy as np


def path_lookup(origin,dest):
    zones=np.unique(np.r_[origin,dest]);index={int(z):i for i,z in enumerate(zones)}
    oi=np.array([index[int(v)] for v in origin]);di=np.array([index[int(v)] for v in dest])
    assert (oi<di).all() and len(oi)==len(zones)*(len(zones)-1)//2
    lookup=np.full((len(zones),len(zones)),-1,int);lookup[oi,di]=np.arange(len(oi))
    return lookup


def draw_cycles(lookup,number,rng):
    parts=[];remaining=number
    while remaining:
        z=np.sort(rng.integers(0,len(lookup),size=(max(2*remaining,32),4)),axis=1)
        z=z[(np.diff(z,axis=1)>0).all(axis=1)][:remaining]
        ids=np.column_stack([lookup[z[:,0],z[:,2]],lookup[z[:,0],z[:,3]],lookup[z[:,1],z[:,2]],lookup[z[:,1],z[:,3]]])
        assert (ids>=0).all()
        parts.append(ids);remaining-=len(ids)
    return np.vstack(parts)


def refine_cycles(f0,target,b,cycles):
    f=np.array(f0,copy=True);sign=np.sign(f0-b)
    sign[np.abs(f0-b)<=1e-10*max(1.,float(np.max(f0)),float(np.max(b)))]=0
    lower=np.where(sign>0,b,0.);upper=np.where(sign<0,b,np.inf)
    accepted=eligible=0;epsilon=1e-12*max(1.,float(np.mean(f0)))
    initial=float(np.abs(f-target).sum())
    for ids in cycles:
        a,c,e,g=ids;s=sign[ids]
        if (s==0).any() or s[0]-s[1]-s[2]+s[3]!=0:continue
        eligible+=1
        v0,v1,v2,v3=f[ids]
        lo=max(lower[a]-v0,v1-upper[c],v2-upper[e],lower[g]-v3)
        hi=min(upper[a]-v0,v1-lower[c],v2-lower[e],upper[g]-v3)
        if lo>hi+epsilon:raise ArithmeticError('Cycle lost feasible zero step')
        residuals=sorted([target[a]-v0,v1-target[c],v2-target[e],target[g]-v3])
        theta=max(lo,min(hi,max(residuals[1],min(0.,residuals[2]))))
        if abs(theta)<=epsilon:continue
        old=abs(v0-target[a])+abs(v1-target[c])+abs(v2-target[e])+abs(v3-target[g])
        new=abs(v0+theta-target[a])+abs(v1-theta-target[c])+abs(v2-theta-target[e])+abs(v3+theta-target[g])
        if new>=old-epsilon:continue
        f[a]+=theta;f[c]-=theta;f[e]-=theta;f[g]+=theta;accepted+=1
    final=float(np.abs(f-target).sum())
    assert final<=initial+1e-8 and f.min()>=-1e-9
    assert abs(float(np.abs(f-b).sum()-np.abs(f0-b).sum()))<1e-7
    return f,dict(cycles_drawn=len(cycles),eligible_cycles=eligible,accepted_updates=accepted,target_L1_before=initial,target_L1_after=final,relative_L1_movement=float(np.abs(f-f0).sum()/max(f0.sum(),1e-12)))
