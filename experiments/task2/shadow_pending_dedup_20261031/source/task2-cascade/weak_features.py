"""Observation-aware slow approach, recovery and forecast uncertainty."""
import numpy as np

def weak_features(s):
    x=np.asarray(s['x'],float);_,n,_=x.shape
    good=(x[:,:,3]>=.75)&(x[:,:,7]>.5)
    names=[];cols=[];models={}
    def add(k,v):names.append(k);cols.append(v)
    def line(v,m,start):
        t=np.arange(13,dtype=float)[:,None]-12;m=m.copy();m[:start]=False
        w=m.astype(float);cnt=w.sum(0);den=np.maximum(cnt,1);tm=(t*w).sum(0)/den;vm=np.where(m,v,0).sum(0)/den
        var=(w*(t-tm)**2).sum(0);cov=np.where(m,(t-tm)*(v-vm),0).sum(0);slope=np.divide(cov,var,out=np.zeros(n),where=var>0)
        now=vm-slope*tm;resid=np.where(m,v-(now+slope*t),0);noise=np.sqrt((resid**2).sum(0)/np.maximum(cnt-2,1))
        now=np.where(cnt>0,now,0);return now,slope,noise,cnt,var
    for c,label in enumerate(['speed','flow','occupancy']):
        v=x[:,:,c];m=good&(x[:,:,4+c]<.5)&np.isfinite(v)
        fits=[]
        for start,span in [(0,60),(6,30)]:
            a,b,noise,cnt,var=line(v,m,start);fits.append((a,b,noise,cnt,var))
            for suffix,value in [('intercept',a),('slope_per5min',b),('residual_noise',noise),('observations',cnt/13),('time_spread',var/182)]:add(f'{label}_fit{span}_{suffix}',value)
        a,b,noise,cnt,var=fits[0];aa,bb,nn,cc,vv=fits[1]
        add(label+'_trend_acceleration',bb-b);add(label+'_recent_vs_long_intercept',aa-a)
        latest=np.zeros(n);first=np.zeros(n);age=np.full(n,13.);first_t=np.zeros(n);last_t=np.zeros(n);seen=np.zeros(n,bool)
        for t in range(13):
            fresh=m[t]&~seen;first[fresh]=v[t,fresh];first_t[fresh]=t
            latest[m[t]]=v[t,m[t]];age[m[t]]=12-t;last_t[m[t]]=t;seen|=m[t]
        med=np.nanmedian(np.where(m,v,np.nan),axis=0);med=np.nan_to_num(med)
        add(label+'_last_minus_median',latest-med);add(label+'_last_minus_long_fit',latest-a);add(label+'_net_observed_change',latest-first);add(label+'_observed_span',(last_t-first_t)/12)
        models[label]=(fits,latest,age)
    speed,latest,age=models['speed'];occ,olast,oage=models['occupancy'];flow,flast,fage=models['flow'];cut=s['network']['threshold']/s['network']['free_speed'];h=np.arange(1,7)[None,:]
    for k,span in enumerate([60,30]):
        a,b,noise,cnt,var=speed[k];oa,ob,onoise,ocnt,ovar=occ[k]
        add(f'fit{span}_slowdown_occupancy_growth',np.maximum(-b,0)*np.maximum(ob,0))
        add(f'fit{span}_speed_occupancy_slope_product',-b*ob)
    link=np.column_stack(cols);future=[];fn=[]
    for k,span in enumerate([60,30]):
        a,b,noise,cnt,var=speed[k]
        forecast=a[:,None]+b[:,None]*h-cut[:,None]
        uncertainty=noise[:,None]*np.sqrt(1+1/np.maximum(cnt[:,None],1)+h*h/np.maximum(var[:,None],1))
        for suffix,value in [('margin',forecast),('uncertainty',uncertainty),('standardized_margin',np.clip(forecast/np.maximum(uncertainty,.01),-30,30)),('from_last_observation',latest[:,None]+b[:,None]*(h+age[:,None])-cut[:,None])]:
            fn.append(f'fit{span}_future_{suffix}');future.append(value)
    future.append((h+age[:,None])/18);fn.append('forecast_distance_from_last_observation')
    a=np.concatenate([np.tile(link[:,None,:],(1,6,1)),np.stack(future,axis=2)],axis=2).astype('float32')
    assert np.isfinite(a).all();return a,names+fn
