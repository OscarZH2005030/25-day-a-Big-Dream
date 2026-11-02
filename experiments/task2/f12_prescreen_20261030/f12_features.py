"""F1/F2 link-level onset features, computed only from s['x'] (13 history frames, T-60..T) and static network."""
import numpy as np
F1_NAMES=['rT','s5','s15','s30','accel','tau','tau_valid','nobs']
F2_NAMES=['rnT','s15n','tau_n','cnt01_n','diff_n','n_nbr']
def _med3(r):
    # r: (13,n) with NaN for missing; median of last <=3 frames ending at each frame
    out=np.full_like(r,np.nan)
    for t in range(r.shape[0]):
        w=r[max(0,t-2):t+1];ok=np.isfinite(w).any(0)
        with np.errstate(all='ignore'):
            m=np.nanmedian(np.where(np.isfinite(w),w,np.nan),axis=0)
        out[t]=np.where(ok,m,np.nan)
    return out
def _slopes(rt):
    s5=rt[12]-rt[11];s15=(rt[12]-rt[9])/3;s30=(rt[12]-rt[6])/6;return s5,s15,s30
def _tau(rT,s15):
    with np.errstate(all='ignore'):
        t=np.where((s15<0)&(rT>0),np.clip(rT/np.maximum(-s15,1e-6),1,12),np.nan)
    return t
def link_features(x,net):
    ratio=x[:,:,0].astype(np.float64);miss=x[:,:,4]>0.5;thr=float(np.nanmedian(net['threshold']/net['free_speed']))
    r=np.where(miss,np.nan,ratio-thr);rt=_med3(r);s5,s15,s30=_slopes(rt);tau=_tau(rt[12],s15)
    F1=np.column_stack([rt[12],s5,s15,s30,s5-s15,tau,np.isfinite(tau).astype(float),(~miss).sum(0).astype(float)])
    n=r.shape[1];nbr=[[] for _ in range(n)];e=np.asarray(net['edge_index'])
    for a,b in e.T:
        if a!=b:nbr[a].append(b);nbr[b].append(a)
    nbr=[sorted(set(v)) for v in nbr];rn=np.full_like(rt,np.nan);nb_mean=np.full(n,np.nan)
    for i in range(n):
        if not nbr[i]:rn[:,i]=rt[:,i];continue
        vals=np.concatenate([rt[:,[i]]]*2+[rt[:,nbr[i]]],axis=1)
        with np.errstate(all='ignore'):rn[:,i]=np.nanmedian(vals,axis=1)
        with np.errstate(all='ignore'):nb_mean[i]=np.nanmean(rt[12,nbr[i]])
    s5n,s15n,s30n=_slopes(rn);taun=_tau(rn[12],s15n);cnt=((rn>=0)&(rn<=0.1)).sum(0).astype(float)
    F2=np.column_stack([rn[12],s15n,taun,cnt,rt[12]-nb_mean,np.array([len(v) for v in nbr],float)])
    return F1.astype(np.float32),F2.astype(np.float32)
