"""Causal road fields and fixed quantile-CDF decoding for ongoing."""
import numpy as np
PANELS=['D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S']
def project(values,valid,basis):
    valid=np.asarray(valid,bool)&np.isfinite(values)
    if not valid.any():return np.full(8,np.nan)
    a=basis[valid];return np.linalg.solve(a.T@a+1e-3*np.eye(8),a.T@values[valid])
def spatial_features(s,basis):
    x=np.asarray(s['x']);n=x.shape[1];out={};times=np.arange(13)*5-60
    for ch,missing,name in [(0,4,'speed'),(1,5,'load'),(2,6,'occupancy')]:
        good=(x[:,:,missing]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);latest=np.full(n,np.nan);age=np.full(n,65.)
        for t in range(13):latest=np.where(good[t],x[t,:,ch],latest);age=np.where(good[t],-times[t],age)
        valid=np.isfinite(latest)&(age<=15);change_valid=valid&good[6];delta=latest-x[6,:,ch]
        for label,values,mask in [('level',latest,valid),('change_from_T30',delta,change_valid)]:
            coef=project(values,mask,basis)
            for j in range(8):out[f'road_{name}_{label}_position_{j}']=np.full(n,coef[j])
            out[f'road_{name}_{label}_coverage']=np.full(n,mask.mean())
        vals=latest[valid];quant=np.quantile(vals,[.1,.5,.9]) if len(vals) else np.full(3,np.nan)
        for label,value in zip(['q10','q50','q90'],quant):out[f'road_{name}_{label}']=np.full(n,value)
    for p in PANELS:out[f'panel_{p}']=np.full(n,float(s['panel']==p))
    a=np.column_stack(list(out.values())).astype('float32');assert not np.isinf(a).any();return a,list(out)
def cdf_zero(quantiles):
    """P(speed margin<=0) from .10/.50/.90 quantiles, with fixed exponential tails."""
    q=np.sort(np.asarray(quantiles,float),axis=-1);a,b,c=q[...,0],q[...,1],q[...,2];span=np.maximum(c-a,.02)
    low=.1*np.exp(np.clip(-a/span,-50,0));high=1-.1*np.exp(np.clip(c/span,-50,0));midlow=.1+.4*np.clip((-a)/np.maximum(b-a,1e-8),0,1);midhigh=.5+.4*np.clip((-b)/np.maximum(c-b,1e-8),0,1)
    p=np.where(0<a,low,np.where(0<b,midlow,np.where(0<c,midhigh,high)));return np.clip(p,1e-6,1-1e-6).astype('float32')
def odds_mix(parent,new):
    a=np.clip(parent,1e-6,1-1e-6);b=np.clip(new,1e-6,1-1e-6);z=.5*(np.log(a)-np.log1p(-a)+np.log(b)-np.log1p(-b));return np.exp(-np.logaddexp(0.,-z)).astype('float32')
