"""Rich features aligned with the team's strongest Transformer time80/20 split.

Historical profiles use June 1-28 only. Optimization uses June 29-December 13;
development December 14-January 4, holdout January 5-February 28.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

FEATURES = ['initial_speed','initial_flow_lane','history_speed','history_flow_lane',
 'visible_speed','visible_flow_lane','visible_mask','quality','previous_speed','next_speed',
 'previous_flow_lane','next_flow_lane','minutes_before','minutes_after','gap_span',
 'has_previous','has_next','up_speed','up_flow_lane','up_valid','down_speed','down_flow_lane',
 'down_valid','up2_speed','up2_flow_lane','down2_speed','down2_flow_lane',
 'on_ramp_flow_lane','off_ramp_flow_lane','on_ramp_valid','off_ramp_valid',
 'free_speed','capacity_lane','lanes','length','time_sin','time_cos','weekday_sin','weekday_cos',
 'position','global_speed','global_flow_lane','global_available','raw_speed','raw_flow_lane',
 'relative_speed','relative_flow','history_support']


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def graph(root, links):
    topology = pd.read_csv(root/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
    positions = {link:i for i,link in enumerate(links)}
    result = []
    for column in ['incoming_link_ids','outgoing_link_ids']:
        neighbors = []
        for link in links:
            text = topology.loc[link,column]
            neighbors.append([positions[v] for v in str(text).split(';') if v in positions] if pd.notna(text) else [])
        k = max(1,max(map(len,neighbors)))
        index = np.zeros((len(links),k), dtype=np.int64)
        weight = np.zeros_like(index,dtype=np.float32)
        for i, values in enumerate(neighbors):
            if values:
                index[i,:len(values)] = values
                weight[i,:len(values)] = 1/len(values)
        result.extend([index,weight])
    return result


def context(values, valid, history):
    t,n = valid.shape
    ticks = np.broadcast_to(np.arange(t)[:,None], (t,n))
    before = np.maximum.accumulate(np.where(valid,ticks,-1),axis=0)
    after = np.minimum.accumulate(np.where(valid,ticks,t)[::-1],axis=0)[::-1]
    before = np.concatenate([np.full((1,n),-1),before[:-1]],axis=0)
    after = np.concatenate([after[1:],np.full((1,n),t)],axis=0)
    has_b,has_a = before>=0,after<t
    bdist,adist = ticks-before,after-ticks
    nodes = np.arange(n)[None,:]
    prev = values[np.clip(before,0,t-1),nodes]
    nxt = values[np.clip(after,0,t-1),nodes]
    prev = np.where(has_b[...,None],prev,history)
    nxt = np.where(has_a[...,None],nxt,history)
    span = bdist+adist
    both = has_b & has_a & (span<=12)
    interpolated = (prev*adist[...,None]+nxt*bdist[...,None])/np.maximum(span[...,None],1)
    initial = np.where(both[...,None],interpolated,history)
    initial = np.where((~both & has_b & (bdist<=3))[...,None],prev,initial)
    initial = np.where((~both & has_a & (adist<=3))[...,None],nxt,initial)
    bdist = np.where(has_b,bdist,48);adist=np.where(has_a,adist,48)
    return initial,prev,nxt,np.minimum(bdist,48)/24,np.minimum(adist,48)/24,np.minimum(bdist+adist,96)/48,has_b,has_a


def neighbor(values, valid, index, weight):
    weights = weight[None,:,:]*valid[:,index]
    denom = weights.sum(-1)
    output = (values[:,index,:]*weights[...,None]).sum(2)/np.maximum(denom[...,None],1e-8)
    return output,denom>0


def ramps(path, mapping, links, lanes):
    n=len(links);index={v:i for i,v in enumerate(links)}
    output=np.zeros((288,n,4),np.float32)
    frame=pd.read_parquet(path)
    frame=frame.merge(mapping,on=['ramp_link_id','ramp_type'],how='left',validate='many_to_one')
    ts=pd.to_datetime(frame.timestamp,utc=True)
    frame['tick']=ts.dt.hour*12+ts.dt.minute//5
    frame['node']=frame.nearest_mainline_link_id.map(index)
    for typ,channel in [('OR',0),('FR',1)]:
        sub=frame[(frame.ramp_type==typ)&frame.node.notna()].copy()
        sub['valid']=sub.flow_vph.notna()&(sub.pct_observed>=75)
        sub['value']=sub.flow_vph.where(sub.valid,0)
        summed=sub.groupby(['tick','node']).agg(value=('value','sum'),valid=('valid','mean'))
        if len(summed):
            ti=summed.index.get_level_values(0).to_numpy(int);ni=summed.index.get_level_values(1).to_numpy(int)
            output[ti,ni,channel]=summed.value.to_numpy()/lanes[ni]/2000
            output[ti,ni,channel+2]=summed.valid.to_numpy()
    return output


def prepare(args):
    meta=json.loads((args.old_cache/'metadata.json').read_text())
    args.cache.mkdir(parents=True,exist_ok=True)
    manifest=dict(features=FEATURES,fit_days=168,eval_days=22,
                  history_dates=['2030-06-01','2030-06-28'],fit_dates=['2030-06-29','2030-12-13'],
                  eval_dates=['2030-12-14','2031-01-04'],holdout_dates=['2031-01-05','2031-02-28'],
                  historical_profile='June 1-28 only; all later labels excluded',panels=[])
    old_repo=args.reference
    references=pd.read_csv(old_repo)
    references=references[references.method=='graph_transformer']
    for p in meta['panels']:
        name=p['panel'];dest=args.cache/name;dest.mkdir(exist_ok=True)
        if (dest/'COMPLETE.json').exists():
            manifest['panels'].append(json.loads((dest/'metadata.json').read_text()))
            print(json.dumps(dict(panel=name,status='cached')),flush=True);continue
        old=args.old_cache/name
        ox=np.load(old/'x.npy',mmap_mode='r');y=np.load(old/'y.npy',mmap_mode='r');target=np.load(old/'target.npy',mmap_mode='r')
        n=ox.shape[-1];links=p['links'];dates=p['dates'][:273]
        assert dates[195]=='2030-12-13' and dates[196]=='2030-12-14' and dates[217]=='2031-01-04' and dates[218]=='2031-01-05'
        weekdays=np.array([dt.date.fromisoformat(d).weekday() for d in dates])
        sums=np.zeros((7,288,n,2),np.float64);counts=np.zeros((7,288,n,1),np.float64)
        for day in range(28):
            truth=np.asarray(y[day]).transpose(1,2,0)
            good=(ox[day,5]>=.75)&(truth[...,0]>0)&(truth[...,1]>0)
            sums[weekdays[day]]+=truth*good[...,None]
            counts[weekdays[day]]+=good[...,None]
        fallback=sums.sum((0,1))/np.maximum(counts.sum((0,1)),1)
        assert np.isfinite(fallback).all()
        np.savez(dest/'historical-profile.npz',sums=sums,counts=counts,fallback=fallback)
        root=args.release/'corridors'/name
        static=pd.read_csv(root/'network/fd_parameters.csv').set_index('link_id').loc[links]
        lanes=static.lanes.to_numpy(np.float32);free=static.free_speed_kmh.to_numpy(np.float32)
        capacity=static.capacity_vph.to_numpy(np.float32)/lanes
        gi,gw,di,dw=graph(root,links)
        np.savez(dest/'graph.npz',up_index=gi,up_weight=gw,down_index=di,down_weight=dw)
        mapping=pd.read_csv(root/'network/ramp_attachment_map.csv')
        rampfiles={v.stem[-10:].replace('_','-'):v for v in (root/'train/ramp_states').rglob('*.parquet')}
        x=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float16,shape=(273,len(FEATURES),288,n))
        scales=np.array([4.,2000/600],np.float32)
        for day,date in enumerate(dates):
            history=sums[weekdays[day]].copy();count=counts[weekdays[day]].copy()
            if day<28:
                truth=np.asarray(y[day]).transpose(1,2,0)
                good=(ox[day,5]>=.75)&(truth[...,0]>0)&(truth[...,1]>0)
                history-=truth*good[...,None];count-=good[...,None]
            history=np.where(count>0,history/np.maximum(count,1),fallback).astype(np.float32)
            history_scaled=history/scales
            valid=(ox[day,4]>0)&(ox[day,5]>=.75)
            raw=np.asarray(ox[day,2:4],dtype=np.float32).transpose(1,2,0)
            values=np.where(valid[...,None],raw,0)
            initial,prev,nxt,gb,ga,span,hb,ha=context(values,valid,history_scaled)
            up,um=neighbor(values,valid,gi,gw);down,dm=neighbor(values,valid,di,dw)
            up2,_=neighbor(up,um,gi,gw);down2,_=neighbor(down,dm,di,dw)
            ramp=ramps(rampfiles[date],mapping,links,lanes)
            tod=np.arange(288,dtype=np.float32)[:,None]*2*np.pi/288
            fraction=valid.mean(1,keepdims=True)
            global_values=values.sum(1)/np.maximum(valid.sum(1,keepdims=True),1)
            position=np.cumsum(static.length_km.to_numpy());position/=position[-1]
            channels=[initial[...,0],initial[...,1],history_scaled[...,0],history_scaled[...,1],
                values[...,0],values[...,1],valid,ox[day,5],prev[...,0],nxt[...,0],prev[...,1],nxt[...,1],
                gb,ga,span,hb,ha,up[...,0],up[...,1],um,down[...,0],down[...,1],dm,
                up2[...,0],up2[...,1],down2[...,0],down2[...,1],
                ramp[...,0],ramp[...,1],ramp[...,2],ramp[...,3],
                free/120,capacity/2400,lanes/8,static.length_km.to_numpy()/5,
                np.sin(tod),np.cos(tod),np.sin(weekdays[day]*2*np.pi/7),np.cos(weekdays[day]*2*np.pi/7),
                position,global_values[:,0,None],global_values[:,1,None],fraction,
                raw[...,0],raw[...,1],initial[...,0]*100/free,initial[...,1]*2000/capacity,
                np.minimum(count[...,0],32)/32]
            assert len(channels)==len(FEATURES)
            for channel,values_ in enumerate(channels):x[day,channel]=np.broadcast_to(values_,(288,n))
            assert np.isfinite(x[day]).all()
            assert not valid[target[day]].any(), 'Hidden target was visible'
            if day%70==0:print(json.dumps(dict(panel=name,day=day)),flush=True)
        x.flush()
        for regime in ['R1','R2','R3']:
            days=[d for d in range(218,273) if p['regimes'][d]==regime]
            count_current=int(target[days].sum())
            expected=references[(references.panel==name)&(references.regime==regime)]
            assert len(expected)==1 and int(expected.n.iloc[0])==count_current,(name,regime,count_current)
        pm=dict(panel=name,family=p['family'],dates=dates,regimes=p['regimes'][:273],links=links,
                old_cache=str(old.resolve()),features=len(FEATURES),nodes=n,reference_targets_match=True)
        save(dest/'metadata.json',pm);save(dest/'COMPLETE.json',dict(status='complete'))
        manifest['panels'].append(pm)
        print(json.dumps(dict(panel=name,status='complete')),flush=True)
    save(args.cache/'metadata.json',manifest)
    print(json.dumps(dict(status='all_complete',features=len(FEATURES))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-cache',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);prepare(p.parse_args())
