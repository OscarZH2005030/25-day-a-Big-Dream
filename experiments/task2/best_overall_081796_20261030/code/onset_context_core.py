"""Strictly previous-day spatial priors and link/clock state references."""
import warnings
import numpy as np
import pandas as pd

def nanstat(a,kind='median',axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmedian(a,axis=axis) if kind=='median' else np.nanmean(a,axis=axis)

class PastContext:
    def __init__(self,nodes):
        self.n=nodes;self.last_date='';self.total=np.zeros((2,48,nodes,5));self.square=np.zeros_like(self.total);self.count=np.zeros_like(self.total)
        self.risk_sum=np.zeros(nodes);self.risk_n=np.zeros(nodes);self.clock_sum=np.zeros((8,nodes));self.clock_n=np.zeros_like(self.clock_sum)
    def save(self,path):
        np.savez_compressed(path,total=self.total,square=self.square,count=self.count,risk_sum=self.risk_sum,risk_n=self.risk_n,clock_sum=self.clock_sum,clock_n=self.clock_n,last_date=np.array(self.last_date),nodes=np.array(self.n))
    @classmethod
    def load(cls,path):
        with np.load(path) as z:
            s=cls(int(z['nodes']));s.last_date=str(z['last_date'])
            for key in ['total','square','count','risk_sum','risk_n','clock_sum','clock_n']:setattr(s,key,z[key])
        return s
    def reference(self,date):
        if self.last_date and date<=self.last_date:raise ValueError('Context must be strictly earlier than the feature date')
        cat=int(pd.Timestamp(date).weekday()>=5);n=self.count.sum(0);mu=np.divide(self.total.sum(0),n,out=np.full(n.shape,np.nan),where=n>0);ex2=np.divide(self.square.sum(0),n,out=np.full(n.shape,np.nan),where=n>0)
        mean=(self.total[cat]+5*mu)/(self.count[cat]+5);second=(self.square[cat]+5*ex2)/(self.count[cat]+5);floor=np.array([.03,.05,.02,.02,.02]);sd=np.sqrt(np.maximum(second-mean*mean,floor**2))
        globalrisk=float(self.risk_sum.sum()/self.risk_n.sum()) if self.risk_n.sum()>0 else np.nan
        risk=(self.risk_sum+10*globalrisk)/(self.risk_n+10)
        return mean,sd,n,risk
    def features(self,s,ramps,ramp_counts,reference=None):
        date=s['origin'][:10]
        if self.last_date and date<=self.last_date:raise ValueError('Same-day or future reference leakage')
        mean,sd,n,risk=self.reference(date) if reference is None else reference
        x=s['x'];nodes=x.shape[1];assert nodes==self.n and ramps.shape==(12,nodes,4)
        origin=pd.Timestamp(s['origin']);slot=origin.hour*12+origin.minute//5;slots=np.arange(slot-12,slot+1);assert slots.min()>=0;bins=slots//6
        common=(x[:,:,3]>=.75)&(x[:,:,7]>.5);raw=np.full((13,nodes,5),np.nan)
        for ch,miss in [(0,4),(1,5),(2,6)]:raw[:,:,ch]=np.where(common&(x[:,:,miss]<.5),x[:,:,ch],np.nan)
        for side in [0,1]:
            count=ramp_counts[:,side];coverage=np.divide(ramps[:,:,side+2],count[None],out=np.ones((12,nodes)),where=count[None]>0)
            raw[:12,:,3+side]=np.where((coverage>=.75)|(count[None]==0),ramps[:,:,side]/s['network']['capacity'][None],np.nan)
        z=np.clip((raw-mean[bins])/sd[bins],-8,8);out={};prior={};names=['speed','load','occupancy','onramp','offramp'];latest={};ages={};slopes={}
        for ch,name in enumerate(names):
            a=z[:,:,ch];last=np.full(nodes,np.nan);age=np.full(nodes,65.);support=np.zeros(nodes)
            for t in range(13):
                good=np.isfinite(a[t]);last=np.where(good,a[t],last);age=np.where(good,(12-t)*5,age);support=np.where(good,n[bins[t],:,ch],support)
            latest[name]=last;ages[name]=age;tail=a[-6:];ii,jj=np.triu_indices(6,1);slope=nanstat((tail[jj]-tail[ii])/((jj-ii)*5)[:,None]);slopes[name]=slope
            out[name+'_z_latest']=last;out[name+'_z_recent_mean']=nanstat(a[-3:],'mean');out[name+'_z_30min_median']=nanstat(tail);out[name+'_z_slope']=slope;out[name+'_reference_confidence']=np.minimum(support,30)/30;out[name+'_recent_z_coverage']=np.isfinite(tail).mean(0)
        out['load_occupancy_excess_product']=np.maximum(latest['load'],0)*np.maximum(latest['occupancy'],0)
        out['load_excess_minus_speed_excess']=latest['load']-latest['speed'];out['occupancy_excess_minus_speed_excess']=latest['occupancy']-latest['speed']
        out['occupancy_growth_with_flow_turnover']=np.maximum(slopes['occupancy'],0)*np.maximum(-slopes['load'],0)
        out['net_ramp_excess']=latest['onramp']-latest['offramp'];out['net_ramp_excess_slope']=slopes['onramp']-slopes['offramp']
        a,b=s['network']['edges']
        for direction,src,dst in [('upstream',a,b),('downstream',b,a)]:
            degree=np.bincount(dst,minlength=nodes)
            for name in ['speed','load','occupancy']:
                values=latest[name];good=np.isfinite(values)&(ages[name]<=15);total=np.zeros(nodes);count=np.zeros(nodes);np.add.at(total,dst,np.where(good[src],values[src],0));np.add.at(count,dst,good[src]);v=np.divide(total,count,out=np.full(nodes,np.nan),where=count>0)
                out[direction+'_'+name+'_z_difference']=v-values;out[direction+'_'+name+'_z_coverage']=np.divide(count,degree,out=np.zeros(nodes),where=degree>0)
        clock=slot//36;clockrisk=(self.clock_sum[clock]+20*risk)/(self.clock_n[clock]+20)
        rank=np.full(nodes,np.nan);good=np.isfinite(risk)
        if good.any():
            _,inv,cnt=np.unique(risk[good],return_inverse=True,return_counts=True);rank[good]=(np.cumsum(cnt)-cnt/2)[inv]/good.sum()
        prior['site_event_risk']=risk;prior['site_event_logit']=np.log(np.clip(risk,1e-5,1-1e-5)/(1-np.clip(risk,1e-5,1-1e-5)));prior['clock_event_risk']=clockrisk;prior['clock_minus_site_risk']=clockrisk-risk;prior['site_risk_percentile']=rank;prior['site_prior_confidence']=self.risk_n/(self.risk_n+10);prior['clock_prior_confidence']=self.clock_n[clock]/(self.clock_n[clock]+20);prior['cold_start']=np.full(nodes,float(not good.any()))
        context=np.column_stack(list(out.values())).astype('float32');priors=np.column_stack(list(prior.values())).astype('float32');assert not np.isinf(context).any() and not np.isinf(priors).any()
        return priors,context,list(prior),list(out)
    def update_day(self,date,mainline,ramps,net,ramp_counts,targets):
        if self.last_date and date<=self.last_date:raise ValueError('Daily updates must be strictly chronological')
        common=mainline['eligible']&np.isfinite(mainline['quality'])&(mainline['quality']>=75);channels=[]
        for name,denom in [('speed',net['free_speed']),('flow',net['capacity']),('occupancy',np.ones(self.n))]:channels.append(np.where(common&np.isfinite(mainline[name]),mainline[name]/denom[None],np.nan))
        for side in [0,1]:
            c=ramp_counts[:,side];coverage=np.divide(ramps[:,:,side+2],c[None],out=np.ones((288,self.n)),where=c[None]>0);channels.append(np.where((coverage>=.75)|(c[None]==0),ramps[:,:,side]/net['capacity'][None],np.nan))
        daily=nanstat(np.stack(channels,-1).reshape(48,6,self.n,5),'mean',axis=1);good=np.isfinite(daily);cat=int(pd.Timestamp(date).weekday()>=5);self.total[cat]+=np.where(good,daily,0);self.square[cat]+=np.where(good,daily*daily,0);self.count[cat]+=good
        if targets:
            positives=np.stack([np.asarray(y,bool)&np.asarray(m,bool) for _,y,m in targets]).any(-1);known=np.stack([np.asarray(m,bool).any(-1) for _,y,m in targets]);value=np.divide(positives.sum(0),known.sum(0),out=np.zeros(self.n),where=known.sum(0)>0);active=known.any(0);self.risk_sum+=value;self.risk_n+=active
            for clock in range(8):
                take=[i for i,(slot,_,_) in enumerate(targets) if slot//36==clock]
                if take:
                    pp=positives[take];kk=known[take];self.clock_sum[clock]+=np.divide(pp.sum(0),kk.sum(0),out=np.zeros(self.n),where=kk.sum(0)>0);self.clock_n[clock]+=kk.any(0)
        self.last_date=date
