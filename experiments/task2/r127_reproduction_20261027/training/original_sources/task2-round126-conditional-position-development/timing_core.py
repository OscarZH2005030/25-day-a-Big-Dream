"""Causal timing features and support-preserving front adjustment."""
import numpy as np
from coverage_data import matrix
from run_round2 import PANELS
def unseen_queue(s):
    x=s['x'];cut=s['network']['threshold']/s['network']['free_speed']
    return ~((x[:,:,4]<.5)&(x[:,:,7]>.5)&(x[:,:,0]<=cut)).any(0)
def features(s):
    n=s['x'].shape[1];x=matrix(s)[:n];panel=np.zeros((n,len(PANELS)),dtype='float32');panel[:,PANELS.index(s['panel'])]=1
    result=np.concatenate([x[:,:185],x[:,191:],panel],axis=1);assert result.shape==(n,557) and np.isfinite(result).all();return result
def adjust(s,base,first_prob,alpha):
    if alpha==0:return base.copy()
    base=np.asarray(base,dtype='float32');first_prob=np.asarray(first_prob,dtype='float32');assert first_prob.shape==(base.shape[1],6)
    assert np.allclose(first_prob.sum(1),1,atol=1e-5)
    cdf=np.cumsum(first_prob,axis=1).T;cdf[-1]=1
    # Retain the base's falling tail. This changes the rising front only approximately.
    tail=base/np.maximum(np.maximum.accumulate(base,axis=0),1e-8)
    shape=cdf*tail;shape/=np.maximum(shape.max(0),1e-8);peak=base.max(0)
    proposed=shape*peak;blend=(1-alpha)*base+alpha*proposed
    blend*=peak/np.maximum(blend.max(0),1e-8);blend=np.minimum(blend,peak)
    # Assign the same float32 peak explicitly, preserving unmasked six-step event support.
    blend[np.argmax(blend,axis=0),np.arange(base.shape[1])]=peak
    use=unseen_queue(s);out=base.copy();out[:,use]=blend[:,use]
    assert np.array_equal(out.max(0),base.max(0)) and np.array_equal(out[:,~use],base[:,~use])
    return out
