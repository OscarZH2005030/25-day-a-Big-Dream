"""Screenshot-inspired causal geometry, robust speed, FD and transition prior."""
import warnings
import numpy as np


def safe_median(a,axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);v=np.nanmedian(a,axis=axis)
    return np.nan_to_num(v),np.isfinite(v)

def geometry(s,fd):
    x=np.asarray(s['x'],float);net=s['network'];n=x.shape[1];cut=net['threshold']/net['free_speed'];good=(x[:,:,3]>=.75)&(x[:,:,7]>.5);valid=good&(x[:,:,4]<.5);speed=x[:,:,0];src,dst=net['edges'];adj=[set([i]) for i in range(n)]
    for a,b in zip(src,dst):adj[a].add(b);adj[b].add(a)
    smooth=np.zeros((13,n));sm=np.zeros((13,n),bool);neighbor_count=np.zeros((13,n))
    # Own and immediate released neighbors, same frame; never order-index edges.
    for i,ids in enumerate(adj):
        ids=sorted(ids);smooth[:,i],sm[:,i]=safe_median(np.where(valid[:,ids],speed[:,ids],np.nan),axis=1);neighbor_count[:,i]=valid[:,ids].sum(1)
    cols=[];names=[]
    def add(k,v):cols.append(np.asarray(v,float));names.append(k)
    last=np.ones(n);age=np.full(n,13.);latest_flow=np.zeros(n);latest_occ=np.zeros(n);fvalid=np.zeros(n,bool)
    for t in range(13):
        last[valid[t]]=speed[t,valid[t]];age[valid[t]]=12-t
        f=good[t]&(x[t,:,5]<.5);latest_flow[f]=x[t,f,1];fvalid|=f
        o=good[t]&(x[t,:,6]<.5);latest_occ[o]=x[t,o,2]
    add('raw_distance_actual_queue_line',last-cut);add('raw_distance_ratio_point60',last-.6);add('local_observation_age',age/13);add('neighbor_frame_coverage',sm.mean(0));add('neighbor_count_latest',neighbor_count[-1]/np.maximum([len(v) for v in adj],1));add('smoothed_latest_minus_local',smooth[-1]-last)
    coefficients={}
    for span in [3,6]:
        tt=np.arange(-span,1,dtype=float)*5;design=np.column_stack([np.ones(span+1),tt,tt*tt]);a=np.zeros((n,3));spread=np.zeros(n);err=np.zeros(n)
        for i in range(n):
            ok=sm[-span-1:,i];spread[i]=ok.mean()
            if ok.sum()>=3:
                a[i]=np.linalg.lstsq(design[ok],smooth[-span-1:,i][ok],rcond=None)[0];err[i]=np.sqrt(np.mean((design[ok]@a[i]-smooth[-span-1:,i][ok])**2))
            elif ok.any():a[i,0]=np.mean(smooth[-span-1:,i][ok])
        coefficients[span]=a
        for label,v in [('distance',a[:,0]-cut),('slope_per_minute',a[:,1]),('acceleration',2*a[:,2]),('fit_coverage',spread),('fit_rmse',err)]:add(f'robust{span*5}_{label}',v)
        linear=np.full(n,120.);quadratic=np.full(n,120.);exists=np.zeros(n)
        for i,(a0,a1,a2) in enumerate(a):
            if a0<=cut[i]:linear[i]=quadratic[i]=0;exists[i]=1;continue
            if a1<0:linear[i]=min((cut[i]-a0)/a1,120)
            roots=np.roots([a2,a1,a0-cut[i]]) if abs(a2)>1e-12 else np.array([(cut[i]-a0)/a1]) if a1 else []
            positive=[float(z.real) for z in roots if abs(z.imag)<1e-8 and z.real>=0]
            if positive:quadratic[i]=min(min(positive),120);exists[i]=1
        add(f'linear_eta{span*5}',linear/120);add(f'quadratic_eta{span*5}',quadratic/120);add(f'quadratic_crossing_exists{span*5}',exists)
    flow=latest_flow*net['capacity'];density=flow/np.maximum(last*net['free_speed'],1.);critical=fd['critical_density'];lanes=fd['lanes']
    add('flow_capacity_ratio',latest_flow);add('density_critical_ratio',density/np.maximum(critical,1));add('per_lane_flow_capacity_ratio',(flow/lanes)/(net['capacity']/lanes));add('fd_flow_observed',fvalid);add('speed_falling_times_demand',np.maximum(-coefficients[3][:,1],0)*latest_flow)
    for direction,aa,bb in [('downstream',src,dst),('upstream',dst,src)]:
        med=np.zeros(n);cov=np.zeros(n);minv=np.zeros(n);travel=np.zeros(n)
        for i in range(n):
            ids=bb[aa==i]
            if not len(ids):continue
            mm=valid[-6:,ids];values=np.where(mm,speed[-6:,ids],np.nan);med[i]=safe_median(values.reshape(-1))[0];cov[i]=mm.mean();minv[i]=np.min(np.where(mm,speed[-6:,ids],np.inf)) if mm.any() else 0;travel[i]=np.mean(60*(net['length_km'][i]+net['length_km'][ids])*.5/net['free_speed'][ids])
        add(direction+'_recent_median',med);add(direction+'_recent_minimum',minv);add(direction+'_history_coverage',cov);add(direction+'_freeflow_minutes',travel/30)
    down_latest=np.zeros(n);downcov=np.zeros(n)
    for i in range(n):
        ids=dst[src==i]
        if len(ids):down_latest[i]=np.maximum((cut[ids]-last[ids]),0).mean();downcov[i]=(age[ids]<13).mean()
    add('local_abrupt_drop_minus_downstream_pressure',np.maximum(-coefficients[3][:,1],0)-down_latest);add('shock_downstream_observed',downcov)
    base=np.column_stack(cols);future=[]
    for h in range(1,7):
        t=h*5;a=coefficients[3];b=coefficients[6];future.append(np.column_stack([base,a[:,0]+a[:,1]*t+a[:,2]*t*t-cut,b[:,0]+b[:,1]*t+b[:,2]*t*t-cut]))
    result=np.stack(future,axis=1).astype('float32');assert np.isfinite(result).all();return result,names+['robust15_projected_margin','robust30_projected_margin']
