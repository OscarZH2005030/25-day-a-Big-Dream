"""Probability ranks preserve order under parent calibration changes."""
import numpy as np

def rank(values):
    v=np.asarray(values,float);_,inv,count=np.unique(v,return_inverse=True,return_counts=True);end=np.cumsum(count);start=end-count;mid=(start+end-1)/2
    return mid[inv]/max(len(v)-1,1)

def relative_features(b,positive_threshold=.25):
    b=np.asarray(b,float);h,n=b.shape;active=b>=positive_threshold;allrank=np.column_stack([rank(b[i]) for i in range(6)]);positive=np.zeros((n,6));globalpos=np.zeros((n,6))
    for i in range(6):
        ids=np.flatnonzero(active[i]);positive[ids,i]=rank(b[i,ids])
    allids=np.flatnonzero(active.ravel());flat=np.zeros(b.size);flat[allids]=rank(b.ravel()[allids]);globalpos=flat.reshape(6,n).T
    linkrank=np.vstack([rank(b[:,i]) for i in range(n)]);peakrank=np.tile(rank(b.max(0))[:,None],(1,6));globalrank=rank(b.ravel()).reshape(6,n).T
    # Top-score ties are explicit rather than broken by link index.
    top=(b==b.max(1,keepdims=True)).T;linktop=(b==b.max(0,keepdims=True)).T
    values=np.stack([allrank,positive,globalpos,linkrank,peakrank,globalrank,top,linktop],axis=2).astype('float32')
    return values,['parent_rank_in_horizon','parent_rank_among_queued_horizon','parent_rank_among_queued_window','parent_rank_within_link_horizons','parent_peak_rank_among_links','parent_rank_in_whole_window','parent_tied_highest_in_horizon','parent_tied_highest_within_link']
