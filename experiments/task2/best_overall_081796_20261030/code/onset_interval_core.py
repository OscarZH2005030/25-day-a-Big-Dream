"""Causal fine features and masked two-stage interval likelihood primitives."""
import warnings
import numpy as np
INTERVALS=np.asarray([(a,b) for a in range(6) for b in range(a,6)],int)
PATTERNS=np.asarray([[(a<=h<=b) for h in range(6)] for a,b in INTERVALS],dtype='float32')

def softmax(z):
    z=np.asarray(z,float);z=z-z.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)

def compatible(y,m):return ((PATTERNS[None]==np.asarray(y)[:,None])|~np.asarray(m,bool)[:,None]).all(-1)

def endpoint_allowed(y,m):
    c=compatible(y,m);a=np.stack([np.stack([c[:,INTERVALS[:,k]==h].any(1) for h in range(6)],1) for k in range(2)],1)
    return a,c.any(1)

def endpoint_grad(z,allowed,weight):
    p=softmax(z);q=p*allowed;q/=np.maximum(q.sum(1,keepdims=True),1e-15)
    return ((p-q)*weight[:,None]).astype('float32'),((p*(1-p)+q*(1-q)+1e-6)*weight[:,None]).astype('float32')

def event_grad(z,conditional_compatibility,known_positive,weight):
    p=np.exp(-np.logaddexp(0.,-np.asarray(z,float)));s=np.asarray(conditional_compatibility,float)
    posterior=np.where(known_positive,1.,p*s/np.maximum(1-p+p*s,1e-15))
    return ((p-posterior)*weight).astype('float32'),((p*(1-p)+posterior*(1-posterior)+1e-6)*weight).astype('float32')

def interval_distribution(start,end):
    z=np.log(np.maximum(np.asarray(start)[:,INTERVALS[:,0]],1e-300))+np.log(np.maximum(np.asarray(end)[:,INTERVALS[:,1]],1e-300))
    return softmax(z)

def interval_decode(probability,threshold):
    """Maximize sum(p-threshold) over contiguous intervals or empty output."""
    utility=(np.asarray(probability)-threshold)@PATTERNS.T;best=utility.argmax(1);gain=utility[np.arange(len(best)),best]
    return PATTERNS[best].astype(bool)&(gain>0)[:,None]

def fine_features(s,ramps,ramp_counts):
    """Reliable paired slopes, acceleration, leading demand, actual directed edges.

    s contains public history through masked T; ramps is exactly T-60..T-5.
    Future labels/masks and future data are never accessed.
    """
    x=np.asarray(s['x']);n=x.shape[1];out={};times=np.arange(13)*5-60
    def add(name,value):out[name]=np.asarray(value,dtype='float32')
    def median(a):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);return np.nanmedian(a,axis=0)
    def avg(a):
        good=np.isfinite(a);return np.divide(np.nansum(a,axis=0),good.sum(0),out=np.full(a.shape[1:],np.nan),where=good.sum(0)>0)
    levels={};ages={};slopes={};series={}
    reliable=(x[:,:,3]>=.75)&(x[:,:,7]>.5)
    for ch,missing,label in [(0,4,'speed'),(1,5,'load'),(2,6,'occupancy')]:
        good=reliable&(x[:,:,missing]<.5)&np.isfinite(x[:,:,ch]);a=np.where(good,x[:,:,ch],np.nan);series[label]=a
        last=np.full(n,np.nan);age=np.full(n,65.)
        for t in range(13):last=np.where(good[t],a[t],last);age=np.where(good[t],-times[t],age)
        levels[label]=last;ages[label]=age;add(label+'_latest',last);add(label+'_age_minutes',age)
        for length in [3,6,12]:
            z=a[-length:];ii,jj=np.triu_indices(length,1);pair=(z[jj]-z[ii])/((jj-ii)*5)[:,None];slope=median(pair);slopes[(label,length)]=slope
            med=median(z);add(f'{label}_reliable_fraction_{length}',np.isfinite(z).mean(0));add(f'{label}_slope_{length}',slope);add(f'{label}_slope_MAD_{length}',median(np.abs(pair-slope)))
            add(f'{label}_level_MAD_{length}',median(np.abs(z-med)));add(f'{label}_last_minus_median_{length}',last-med)
            add(f'{label}_decreasing_pair_fraction_{length}',avg(np.where(np.isfinite(pair),(pair<0).astype(float),np.nan)))
        add(label+'_slope_acceleration',slopes[(label,3)]-slopes[(label,6)])
    cut=np.asarray(s['network']['threshold'])/np.asarray(s['network']['free_speed']);margin=levels['speed']-cut
    add('reliable_speed_margin',margin);add('margin_projected_5',margin+slopes[('speed',6)]*(ages['speed']+5));add('margin_projected_30',margin+slopes[('speed',6)]*(ages['speed']+30))
    add('time_to_threshold',np.divide(margin,-slopes[('speed',6)],out=np.full(n,np.nan),where=(slopes[('speed',6)]<-1e-6)&(margin>0))-ages['speed'])
    for lag in [0,1,2]:
        dv=np.diff(series['speed'],axis=0);dq=np.diff(series['load'],axis=0);do=np.diff(series['occupancy'],axis=0)
        v=dv[-6:];q=dq[-6-lag:len(dq)-lag if lag else None];o=do[-6-lag:len(do)-lag if lag else None]
        for label,z in [('load',q),('occupancy',o)]:
            known=np.isfinite(v)&np.isfinite(z);add(f'{label}_rise_precedes_speed_drop_lag{lag}',avg(np.where(known,((z>0)&(v<0)).astype(float),np.nan)));add(f'{label}_lead_pair_coverage_lag{lag}',known.mean(0))
    src,dst=s['network']['edges']
    for direction,source,target in [('upstream',src,dst),('downstream',dst,src)]:
        degree=np.bincount(target,minlength=n);add(direction+'_released_degree',degree)
        for field,values in [('speed',levels['speed']),('load',levels['load']),('speed_slope',slopes[('speed',6)]),('load_slope',slopes[('load',6)])]:
            age=ages['speed' if field.startswith('speed') else 'load'];good=np.isfinite(values)&(age<=15);total=np.zeros(n);count=np.zeros(n)
            np.add.at(total,target,np.where(good[source],values[source],0));np.add.at(count,target,good[source]);mean=np.divide(total,count,out=np.full(n,np.nan),where=count>0)
            add(direction+'_'+field+'_minus_own',mean-values);add(direction+'_'+field+'_coverage',np.divide(count,degree,out=np.zeros(n),where=degree>0))
    cap=np.asarray(s['network']['capacity']);ramp_series=[]
    assert ramps.shape==(12,n,4)
    for side,label in [(0,'onramp'),(1,'offramp')]:
        count=ramp_counts[:,side];coverage=np.divide(ramps[:,:,side+2],count[None],out=np.ones((12,n)),where=count[None]>0);z=np.where((coverage>=.75)|(count[None]==0),ramps[:,:,side]/cap[None],np.nan);ramp_series.append(z)
        add(label+'_latest_load',z[-1]);add(label+'_recent_coverage',(coverage[-6:]>=.75).mean(0));add(label+'_15min_change',z[-1]-z[-4]);add(label+'_30min_change',z[-1]-z[-7])
    net=ramp_series[0]-ramp_series[1];add('ramp_net_latest',net[-1]);add('ramp_net_15min_change',net[-1]-net[-4]);add('ramp_net_30min_change',net[-1]-net[-7])
    values=np.column_stack(list(out.values())).astype('float32');assert not np.isinf(values).any();return values,list(out)
