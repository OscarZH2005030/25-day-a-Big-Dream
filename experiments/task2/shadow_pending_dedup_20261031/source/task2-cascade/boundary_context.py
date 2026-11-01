"""Directed queue-front evidence and coherence of the parent's forecast."""
import numpy as np

def directed_features(s,b):
    x=np.asarray(s['x'],float);n=x.shape[1];cut=s['network']['threshold']/s['network']['free_speed'];good=(x[:,:,3]>=.75)&(x[:,:,7]>.5);valid=good&(x[:,:,4]<.5);q=valid&(x[:,:,0]<=cut);last=np.zeros(n,bool);seen=np.zeros(n,bool);age=np.full(n,13.);speed=np.ones(n);first=np.ones(n);oldq=np.zeros(n,bool)
    for t in range(13):
        fresh=valid[t]&~seen;first[fresh]=x[t,fresh,0];seen|=valid[t];last[valid[t]]=q[t,valid[t]];speed[valid[t]]=x[t,valid[t],0];age[valid[t]]=12-t
        if t<=6:oldq[valid[t]]=q[t,valid[t]]
    names=[];cols=[];src,dst=s['network']['edges'];degrees={};latest={};hist={}
    def add(name,v):names.append(name);cols.append(v)
    for direction,sources,targets in [('upstream',src,dst),('downstream',dst,src)]:
        degree=np.bincount(targets,minlength=n).astype(float);degree_safe=np.maximum(degree,1);degrees[direction]=degree;known=np.zeros(n);queued=np.zeros(n);clear=np.zeros(n);stale=np.zeros(n);recover=np.zeros(n);growth=np.zeros(n);margin=np.zeros(n);joint=np.zeros(n)
        hist_known=np.zeros((13,n));hist_queue=np.zeros((13,n))
        for a,j in zip(sources,targets):
            known[j]+=seen[a];queued[j]+=seen[a]&last[a];clear[j]+=seen[a]&~last[a];stale[j]+=age[a]/13;recover[j]+=seen[a]&(speed[a]>first[a]);growth[j]+=int(last[a])-int(oldq[a]);margin[j]+=np.where(seen[a],speed[a]-cut[a],0);hist_known[:,j]+=valid[:,a];hist_queue[:,j]+=q[:,a]
        for label,value in [('known_share',known/degree_safe),('queue_share',queued/degree_safe),('clear_share',clear/degree_safe),('mean_observation_age',stale/degree_safe),('recovering_share',recover/degree_safe),('queue_growth',growth/degree_safe),('speed_margin',margin/degree_safe),('no_neighbors',degree==0)]:add(direction+'_'+label,value)
        latest[direction]=(known,queued,clear);hist[direction]=(hist_known,hist_queue)
        for span in [3,6,12]:
            k=hist_known[-span:];qq=hist_queue[-span:];add(direction+f'_history_queue_share_{span}',np.divide(qq.sum(0),k.sum(0),out=np.zeros(n),where=k.sum(0)>0));add(direction+f'_history_coverage_{span}',k.sum(0)/degree_safe/span)
    up,down=latest['upstream'],latest['downstream'];add('local_queued_upstream_clear',last&(up[2]>0));add('local_queued_downstream_clear',last&(down[2]>0));add('local_queued_both_sides_clear',last&(up[2]>0)&(down[2]>0));add('local_clear_downstream_queue',seen&~last&(down[1]>0));add('local_clear_upstream_queue',seen&~last&(up[1]>0));add('local_queued_two_sided_queue',last&(up[1]>0)&(down[1]>0));add('neighbor_queue_direction_difference',down[1]/np.maximum(degrees['downstream'],1)-up[1]/np.maximum(degrees['upstream'],1))
    # Same-time observed neighbour changes, not stale values interpolated as
    # a propagating wave. Unobserved transitions have explicit zero coverage.
    for direction in ['upstream','downstream']:
        hk,hq=hist[direction];share=np.divide(hq,hk,out=np.zeros_like(hq),where=hk>0);pairs=valid[1:]&valid[:-1]&(hk[1:]>0)&(hk[:-1]>0);neighborchange=share[1:]-share[:-1];localchange=q[1:].astype(float)-q[:-1].astype(float)
        for span in [3,6,12]:
            m=pairs[-span:];count=m.sum(0);same=((neighborchange[-span:]*localchange[-span:])>0)&m
            add(direction+f'_joint_transition_{span}',np.divide(same.sum(0),count,out=np.zeros(n),where=count>0));add(direction+f'_transition_coverage_{span}',count/span)
    link=np.column_stack(cols);parent=b>=.25;future=[];fn=[]
    def addh(name,value):fn.append(name);future.append(value)
    upcount=np.zeros_like(b,dtype=float);downcount=np.zeros_like(b,dtype=float);upknown=np.maximum(degrees['upstream'],1);downknown=np.maximum(degrees['downstream'],1)
    for a,j in zip(src,dst):upcount[:,j]+=parent[:,a];downcount[:,a]+=parent[:,j]
    upshare=(upcount/upknown).T;downshare=(downcount/downknown).T;addh('parent_upstream_queue_share',upshare);addh('parent_downstream_queue_share',downshare);addh('parent_internal_queue_bridge',(upcount.T>0)&(downcount.T>0));addh('parent_predicted_isolated_queue',parent.T&(upcount.T+downcount.T==0));addh('parent_queue_front_direction',downshare-upshare)
    prev=np.vstack([parent[:1],parent[:-1]]).T;nxt=np.vstack([parent[1:],parent[-1:]]).T;addh('parent_queue_both_adjacent_horizons',prev&nxt);addh('parent_queue_temporal_start',parent.T&~prev);addh('parent_queue_temporal_end',parent.T&~nxt);addh('parent_later_queue_share',np.column_stack([parent[h:].mean(0) for h in range(6)]));addh('parent_earlier_queue_share',np.column_stack([parent[:h+1].mean(0) for h in range(6)]))
    a=np.concatenate([np.tile(link[:,None,:],(1,6,1)),np.stack(future,axis=2)],axis=2).astype('float32');assert np.isfinite(a).all();return a,names+fn
