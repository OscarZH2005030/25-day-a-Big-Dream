"""Causal calendar/structural-region features and leave-day-out fit profiles."""
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

def time_band(slots):
    hour=(np.asarray(slots)%288)/12
    return np.select([hour<6,hour<10,hour<15,hour<19],[0,1,2,3],default=4).astype(int)

def subtract_day(stats,day_stats,weekday):
    result=stats.copy();result[weekday]-=day_stats
    assert result[...,0].min()>-1e-8
    result[...,0]=np.maximum(result[...,0],0)
    return result

class Context:
    def __init__(self,args,backend,root,read,write,progress):
        self.args=args;self.parent,self.sample,self.base,self.nets,*_=backend
        self.root=Path(root);self.read=read;self.write=write;self.progress=progress;self.loaded={};self.geo={}
        self.root.mkdir(exist_ok=True)

    def build_profiles(self,entries):
        by=defaultdict(set)
        for e in entries:by[e['panel']].add(e['origin'][:10])
        for panel,dates in by.items():
            dest=self.root/(panel+'.npz')
            if dest.exists():continue
            net=self.nets[panel];n=len(net['link_ids']);total=np.zeros((7,5,n,4));early=total.copy();parts=[];hashes=[]
            dates=sorted(dates)
            for i,d in enumerate(dates):
                v=self.sample.load_day(panel,d);valid=v['eligible']&np.isfinite(v['quality'])&(v['quality']>=75)&np.isfinite(v['speed'])&np.isfinite(v['flow'])
                flow=np.clip(np.nan_to_num(v['flow']/net['capacity'],nan=0),0,4);queue=(v['speed']<=net['threshold'])&valid;ds=np.zeros((5,n,4));bands=time_band(np.arange(288))
                for b in range(5):
                    m=valid[bands==b];f=flow[bands==b];ds[b,:,0]=m.sum(0);ds[b,:,1]=(f*m).sum(0);ds[b,:,2]=(f*f*m).sum(0);ds[b,:,3]=queue[bands==b].sum(0)
                weekday=date.fromisoformat(d).weekday();total[weekday]+=ds
                if d<='2030-09-30':early[weekday]+=ds
                parts.append(ds);hashes.append(v['key'])
                if (i+1)%64==0:self.progress('fit_calendar_profiles',panel=panel,days=i+1,total=len(dates))
            np.savez_compressed(dest,dates=np.array(dates),days=np.stack(parts),early=early,full=total)
            self.write(dest.with_suffix('.json'),dict(panel=panel,dates=dates,day_content_keys=hashes,source='public fit observations',training_current_day_excluded=True,holiday_mapping=None,no_city_label_invented=True))

    def load(self,panel):
        if panel not in self.loaded:
            with np.load(self.root/(panel+'.npz')) as z:self.loaded[panel]={k:z[k] for k in z.files}
            self.loaded[panel]['index']={d:i for i,d in enumerate(self.loaded[panel]['dates'])}
        return self.loaded[panel]

    @lru_cache(maxsize=4)
    def effective(self,panel,d,regime):
        p=self.load(panel);stats=p[regime];cut='2030-09-30' if regime=='early' else '2030-11-30'
        if d<=cut and d in p['index']:stats=subtract_day(stats,p['days'][p['index'][d]],date.fromisoformat(d).weekday())
        return stats,stats.sum((0,1))

    def geometry(self,panel):
        if panel in self.geo:return self.geo[panel]
        net=self.nets[panel];n=len(net['link_ids']);top=pd.read_csv(self.args.data/'corridors'/panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id').reindex(net['link_ids'])
        order=top.order_index.to_numpy(float);known=np.isfinite(order);norm=np.zeros(n);norm[known]=(order[known]-order[known].min())/max(np.ptp(order[known]),1)
        zone=np.minimum((norm*4).astype(int),3);onehot=np.eye(4)[zone];onehot[~known]=0
        edges=net['edges'];ind=np.zeros(n);out=np.zeros(n);drop=np.zeros(n);capdrop=np.zeros(n);neighbors=[set() for _ in range(n)]
        for up,down in edges.T:
            out[up]+=1;ind[down]+=1;neighbors[up].add(int(down));neighbors[down].add(int(up))
            drop[up]=max(drop[up],max(0.,net['static'][up,1]-net['static'][down,1]))
            capdrop[up]=max(capdrop[up],max(0.,1-net['capacity'][down]/net['capacity'][up]))
        size=np.zeros(n);visited=set()
        for start in range(n):
            if start in visited:continue
            seen={start};stack=[start]
            while stack:
                cur=stack.pop()
                for other in neighbors[cur]:
                    if other not in seen:seen.add(other);stack.append(other)
            visited|=seen;size[list(seen)]=len(seen)/n
        count=lambda col:np.array([sum(bool(x) for x in str(v if pd.notna(v) else '').split(';')) for v in top[col]],float)
        a=np.column_stack([norm,~known,onehot,ind/4,out/4,count('on_ramp_link_ids')/4,count('off_ramp_link_ids')/4,drop,capdrop,net['isolated'],size]).astype('float32')
        assert a.shape==(n,14);self.geo[panel]=a;return a

    def lookup(self,stats,global_stats,weekdays,slots):
        n=global_stats.shape[0];bands=time_band(slots)
        a=stats[np.asarray(weekdays)[:,None],bands[:,None],np.arange(n)[None,:]]
        a=np.where((a[...,0]<10)[...,None],global_stats[None],a)
        fallback=global_stats.sum(0);a=np.where((a[...,0]<1)[...,None],fallback[None,None],a)
        count=np.maximum(a[...,0],1);mean=a[...,1]/count;std=np.sqrt(np.maximum(a[...,2]/count-mean*mean,.0025));risk=a[...,3]/count
        return mean,std,risk,count

    def extra(self,s,regime):
        panel=s['panel'];d=s['origin'][:10];weekday=date.fromisoformat(d).weekday();origin=int(s['origin'][11:13])*12+int(s['origin'][14:16])//5;n=s['x'].shape[1]
        stats,global_stats=self.effective(panel,d,regime);past=np.arange(origin-12,origin+1);future=np.arange(origin+1,origin+7)
        mean,std,risk,count=self.lookup(stats,global_stats,(weekday+past//288)%7,past)
        fm,fs,fr,_=self.lookup(stats,global_stats,(weekday+future//288)%7,future)
        x=s['x'];valid=(x[:,:,5]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);cnt=valid.sum(0)
        obs=np.divide(np.where(valid,x[:,:,1],0).sum(0),cnt,out=mean[-1].copy(),where=cnt>0)
        expected=np.divide(np.where(valid,mean,0).sum(0),cnt,out=mean[-1].copy(),where=cnt>0)
        scale=np.maximum(std.mean(0),.05);dev=np.clip((obs-expected)/scale,-5,5)
        profile=np.column_stack([risk[-1],global_stats[:,3]/np.maximum(global_stats[:,0],1),mean[-1],std[-1],expected,dev,np.clip(obs/np.maximum(expected,.05),0,4),np.log1p(count[-1])/10,cnt==0])
        band=int(time_band(origin));clock=origin/12
        common=np.r_[np.eye(5)[band],np.eye(7)[weekday],float(weekday>=5),np.sin(weekday*2*np.pi/7),np.cos(weekday*2*np.pi/7),max(0,1-abs(clock-8)/3),max(0,1-abs(clock-17)/3)]
        extra=np.column_stack([self.geometry(panel),np.tile(common,(n,1)),profile]).astype('float32')
        fband=np.eye(5)[time_band(future)];angle=(future%288)*2*np.pi/288
        time=np.column_stack([fband,np.sin(angle),np.cos(angle)])
        fextra=np.concatenate([np.tile(time[:,None,:],(1,n,1)),fm[:,:,None],fr[:,:,None],fs[:,:,None]],axis=2).transpose(1,0,2).astype('float32')
        assert extra.shape==(n,40) and fextra.shape==(n,6,10) and np.isfinite(extra).all() and np.isfinite(fextra).all()
        return extra,fextra
