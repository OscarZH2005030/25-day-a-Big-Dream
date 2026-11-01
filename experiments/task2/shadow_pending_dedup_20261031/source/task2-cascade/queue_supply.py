"""Observed inflow/outflow pressure; missing ramps never count as zero."""
import numpy as np

def supply_features(s,h,top):
    x=s['x'][:12];n=x.shape[1];cap=s['network']['capacity'];flow=x[:,:,1]*cap[None];own=(x[:,:,5]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);incoming=np.zeros((12,n));known=np.zeros((12,n));degree=np.zeros(n)
    for src,dst in s['network']['edges'].T:
        incoming[:,dst]+=np.where(own[:,src],flow[:,src],0);known[:,dst]+=own[:,src];degree[dst]+=1
    counts=top['counts'];rampok=[];ramp=[]
    for side in [0,1]:
        ok=(h[:,:,side+2]>=counts[None,:,side])|(counts[None,:,side]==0);rampok.append(ok);ramp.append(h[:,:,side])
    internal=(known>=degree[None])&(degree[None]>0);ramps=rampok[0]&rampok[1];valid=own&internal&ramps;external=np.asarray(top['external'])>0;complete=valid&~external[None]
    pressure=(incoming-flow+ramp[0]-ramp[1])/cap[None];speedgood=(x[:,:,4]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);occgood=(x[:,:,6]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);dvs=np.zeros((12,n));dos=np.zeros((12,n));dvs[1:]=x[1:,:,0]-x[:-1,:,0];dos[1:]=x[1:,:,2]-x[:-1,:,2];paired=np.zeros((12,n),bool);paired[1:]=speedgood[1:]&speedgood[:-1]&occgood[1:]&occgood[:-1]
    cols=[];names=[]
    def mean(v,m):return np.divide(np.where(m,v,0).sum(0),m.sum(0),out=np.zeros(n),where=m.sum(0)>0)
    def add(name,v):names.append(name);cols.append(v)
    for span in [3,6,12]:
        m=valid[-span:];p=pressure[-span:];observed=paired[-span:]&m
        add(f'pressure_mean_{span}',mean(p,m));add(f'pressure_positive_mean_{span}',mean(np.maximum(p,0),m));add(f'pressure_negative_mean_{span}',mean(np.maximum(-p,0),m));add(f'pressure_coverage_{span}',m.mean(0));add(f'complete_balance_coverage_{span}',complete[-span:].mean(0))
        add(f'pressure_speed_recovery_occupancy_fall_{span}',mean(np.maximum(-p,0)*np.maximum(dvs[-span:],0)*np.maximum(-dos[-span:],0),observed));add(f'pressure_queue_feed_{span}',mean(np.maximum(p,0)*np.maximum(dos[-span:],0),observed))
        for side,label in [(0,'on'),(1,'off')]:
            add(f'{label}_ramp_mean_{span}',mean(ramp[side][-span:]/cap[None],rampok[side][-span:]));add(f'{label}_ramp_coverage_{span}',rampok[side][-span:].mean(0))
    add('external_inflow_unobserved',external);add('no_internal_inflow_sensor',degree==0);add('on_ramp_count',counts[:,0]/10);add('off_ramp_count',counts[:,1]/10)
    a=np.column_stack(cols).astype('float32');assert np.isfinite(a).all();return np.tile(a[:,None],(1,6,1)),names
