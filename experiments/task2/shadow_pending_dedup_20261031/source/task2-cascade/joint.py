"""Causal, aligned multi-channel evidence; no target or future measurements."""
import numpy as np


def joint_features(s):
    x=np.asarray(s['x'],float);t,n,_=x.shape
    assert t==13
    good=(x[:,:,3]>=.75)&(x[:,:,7]>.5)
    valid=[good&(x[:,:,4+c]<.5)&np.isfinite(x[:,:,c]) for c in range(3)]
    names=[];cols=[];local=[];blocks={}
    def add(name,value):
        names.append(name);cols.append(np.asarray(value));return value
    def mean(a,m):
        return np.divide(np.where(m,a,0).sum(0),m.sum(0),out=np.zeros(n),where=m.sum(0)>0)
    for c,label in enumerate(['speed','flow','occupancy']):
        m=valid[c];age=np.min(np.where(m,12-np.arange(13)[:,None],13),axis=0)
        add(label+'_latest_age',age/13)
        for span in [3,6,12]:add(label+f'_coverage_{span}',m[-span-1:].mean(0))
    for lag in [1,3,6]:
        vs=valid[0][lag:]&valid[0][:-lag]
        vo=valid[2][lag:]&valid[2][:-lag]
        vf=valid[1][lag:]&valid[1][:-lag]
        dec=x[:-lag,:,0]-x[lag:,:,0];rise=x[lag:,:,2]-x[:-lag,:,2];flow=x[lag:,:,1]-x[:-lag,:,1]
        both=vs&vo;allm=both&vf
        # Same timestamps and same link. Invalid observations never contribute.
        dp=np.maximum(dec,0);op=np.maximum(rise,0);joint=dp*op
        for span in [3,6,12]:
            mm=both[-span:];j=mean(joint[-span:],mm);d=mean(np.where(vo,dec,0)[-span:],mm);o=mean(rise[-span:],mm)
            prefix=f'lag{lag}_recent{span}'
            add(prefix+'_paired_deceleration',d);add(prefix+'_paired_occupancy_rise',o)
            add(prefix+'_joint_pressure',j)
            add(prefix+'_discordant_speed_occupancy',mean((dp*np.maximum(-rise,0)+np.maximum(-dec,0)*op)[-span:],mm))
            add(prefix+'_deceleration_flow_rise',mean((dp*np.maximum(flow,0))[-span:],allm[-span:]))
            add(prefix+'_deceleration_flow_fall',mean((dp*np.maximum(-flow,0))[-span:],allm[-span:]))
            add(prefix+'_paired_coverage',mm.mean(0))
        if lag==1:
            local_joint=np.where(both,joint,0);local_dec=np.where(vs,dp,0);local_rise=np.where(vo,op,0);local_mask=both
    local_count=len(cols)
    src,dst=np.asarray(s['network']['edges'],int)
    for delay in [0,1,3]:
        # Downstream evidence may precede the local response. Aggregate each
        # actual edge after multiplying aligned observations, never maxima
        # from different neighbours or different times.
        a=local_joint[delay:];b=local_joint[:len(local_joint)-delay or None]
        d=local_dec[delay:];o=local_rise[delay:]
        am=local_mask[delay:];bm=local_mask[:len(local_mask)-delay or None]
        sums=np.zeros((n,5));counts=np.zeros(n)
        for up,down in zip(src,dst):
            mm=am[:,up]&bm[:,down]
            if not mm.any():continue
            values=[a[:,up]*b[:,down],d[:,up]*b[:,down],o[:,up]*b[:,down],b[:,down],a[:,up]-b[:,down]]
            sums[up]+=np.array([v[mm].mean() for v in values]);counts[up]+=1
        for k,label in enumerate(['coherent_joint','local_deceleration_downstream_joint','local_occupancy_downstream_joint','downstream_joint','local_minus_downstream_joint']):
            add(f'downstream_leads_{delay*5}min_'+label,np.divide(sums[:,k],counts,out=np.zeros(n),where=counts>0))
        add(f'downstream_leads_{delay*5}min_evidence_coverage',counts/np.maximum(np.bincount(src,minlength=n),1))
    # Horizon-specific interactions tell the verifier whether short-lived
    # evidence supports an imminent event or is being stretched far ahead.
    link=np.column_stack(cols).astype('float32');h=np.arange(1,7)/6
    anchor=names.index('lag1_recent3_joint_pressure')
    slow=names.index('lag3_recent6_joint_pressure')
    interaction=np.stack([link[:,anchor,None]*(1-h),link[:,anchor,None]*h,link[:,slow,None]*h],axis=2)
    result=np.concatenate([np.tile(link[:,None],(1,6,1)),interaction],axis=2)
    names+=['recent_joint_times_near_horizon','recent_joint_times_far_horizon','slow_joint_times_far_horizon']
    assert np.isfinite(result).all()
    return result,names,local_count
