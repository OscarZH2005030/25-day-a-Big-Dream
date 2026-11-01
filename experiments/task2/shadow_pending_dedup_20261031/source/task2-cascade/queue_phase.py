"""Observed queue phase and evidence of clearance, not future phase labels."""
from collections import deque
import numpy as np

PHASES=['observed_core','observed_boundary','observed_recovery','unresolved']

def phase_features(s,b):
    x=np.asarray(s['x'],float);n=x.shape[1];net=s['network'];cut=net['threshold']/net['free_speed'];valid=(x[:,:,4]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);queued=valid&(x[:,:,0]<=cut)
    seen=np.zeros(n,bool);lastq=np.zeros(n,bool);latest=np.ones(n);age=np.full(n,13.);afterclear=np.zeros(n);lastqueue=np.full(n,-1.);lastclear=np.full(n,-1.);transitions=np.zeros(n);episodes=np.zeros(n);clear_start=np.full(n,-1.)
    for t in range(13):
        old=lastq.copy();new=queued[t];ok=valid[t];transitions+=ok&seen&(new!=old);episodes+=ok&new&(~seen|~old)
        starts=ok&~new&(~seen|old);clear_start[starts]=t;clear_start[ok&new]=-1
        afterclear=np.where(ok&new,0,np.where(ok,afterclear+1,afterclear));lastqueue[ok&new]=t;lastclear[ok&~new]=t;latest[ok]=x[t,ok,0];age[ok]=12-t;lastq[ok]=new[ok];seen|=ok
    ever=queued.any(0);adj=[set() for _ in range(n)];upq=np.zeros(n);downq=np.zeros(n);upclear=np.zeros(n);downclear=np.zeros(n);upseen=np.zeros(n);downseen=np.zeros(n);updeg=np.zeros(n);downdeg=np.zeros(n)
    for src,dst in net['edges'].T:
        adj[src].add(int(dst));adj[dst].add(int(src));updeg[dst]+=1;downdeg[src]+=1
        upq[dst]+=lastq[src]&seen[src];downq[src]+=lastq[dst]&seen[dst];upclear[dst]+=seen[src]&~lastq[src];downclear[src]+=seen[dst]&~lastq[dst];upseen[dst]+=seen[src];downseen[src]+=seen[dst]
    phase=np.full(n,3,int);phase[seen&lastq]=0;phase[seen&lastq&((upclear+downclear)>0)]=1;phase[seen&~lastq&ever]=2;phase[seen&~lastq&~ever&((upq+downq)>0)]=1
    compsize=np.zeros(n);distance=np.full(n,n+1.,float);visited=set()
    for start in np.flatnonzero(lastq&seen):
        if int(start) in visited:continue
        component={int(start)};stack=[int(start)]
        while stack:
            cur=stack.pop()
            for other in adj[cur]:
                if seen[other] and lastq[other] and other not in component:component.add(other);stack.append(other)
        visited|=component;ids=list(component);compsize[ids]=len(ids)
        frontier=[v for v in ids if upclear[v]+downclear[v]>0];todo=deque(frontier)
        for v in frontier:distance[v]=0
        while todo:
            cur=todo.popleft()
            for other in adj[cur]:
                if other in component and distance[other]>distance[cur]+1:distance[other]=distance[cur]+1;todo.append(other)
    degree=updeg+downdeg;known=upseen+downseen
    col=[seen,lastq,ever,age/13,afterclear/13,(12-lastqueue)/13,(12-lastclear)/13,np.where(clear_start>=0,12-clear_start,0)/13,transitions/12,episodes/13,latest-cut,
         queued[-4:].sum(0)/np.maximum(valid[-4:].sum(0),1),valid[-4:].sum(0)/4,compsize/np.maximum(n,1),np.minimum(distance,6)/6,distance<=n,
         upq/np.maximum(updeg,1),downq/np.maximum(downdeg,1),upclear/np.maximum(updeg,1),downclear/np.maximum(downdeg,1),known/np.maximum(degree,1),degree==0]
    names=['latest_seen','latest_queued','ever_queued','latest_age','observed_clear_count_after_queue','time_since_last_queue','time_since_last_clear','recovery_elapsed','queue_state_transitions','queue_episodes','latest_speed_margin','recent_queue_share','recent_speed_coverage','observed_component_fraction','graph_distance_to_clear_boundary','has_observed_clear_boundary','upstream_queue_share','downstream_queue_share','upstream_clear_share','downstream_clear_share','neighbor_observation_coverage','no_graph_neighbors']
    link=np.column_stack(col);parent=b>=.25;total=parent.sum(1);perlink=parent.sum(0);nlinks=parent.any(0).sum();h=np.arange(6)
    pfeatures=np.stack([np.tile(total,(n,1))/max(n,1),np.tile(1/np.maximum(total,1),(n,1)),np.tile((perlink/6)[:,None],(1,6)),np.tile((parent.any(0)/max(nlinks,1))[:,None],(1,6)),parent.T,np.tile((total.max()-total.min())/max(n,1),(n,6)),np.tile(np.arange(1,7)/6,(n,1))*np.tile((age/13)[:,None],(1,6))],axis=2)
    names+=['parent_window_queue_fraction','inverse_parent_window_queue_cells','parent_link_queue_share','parent_link_fraction_of_event_links','parent_cell_queued','parent_window_queue_size_range','forecast_distance_times_observation_age']
    a=np.concatenate([np.tile(link[:,None,:],(1,6,1)),pfeatures,np.tile(np.eye(4)[phase,None,:],(1,6,1))],axis=2).astype('float32');names+=['phase_'+p for p in PHASES]
    assert np.isfinite(a).all();return a,phase,names

def invariant_parent_features(x):
    """Use the parent's frozen decisions instead of its absolute calibration."""
    out=x.copy();start=x.shape[1]-11
    for j in [0,3,4,5]:out[:,start+j]=(x[:,start+j]>=.25)
    out[:,start+6]=x[:,start+7]
    return out
