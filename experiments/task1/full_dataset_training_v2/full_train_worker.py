"""Joint traffic-state reconstruction with factorized temporal/graph attention.

This is a custom implementation, inspired by attention-based imputation and
traffic graph Transformers; it is NOT a reproduction of SAITS or PDFormer.
Conservation uses topology-derived detector flows, NOT organizer boundary flux.
Its loss is an explicitly approximate regularizer, not the official Task 3 loss.
"""
import argparse
import json
import math
import time
from pathlib import Path
import tempfile
from training_io import RunLock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from task1_baseline import Experiment, CHANNELS, KEYS, score, log


def interpolate(a, historical):
    tick = np.broadcast_to(np.arange(a.shape[1]), a.shape)
    valid = np.isfinite(a)
    left = np.maximum.accumulate(np.where(valid, tick, -1), axis=1)
    right = np.minimum.accumulate(np.where(valid, tick, a.shape[1])[:, ::-1], axis=1)[:, ::-1]
    pv = np.take_along_axis(a, left.clip(0, a.shape[1]-1), axis=1)
    nv = np.take_along_axis(a, right.clip(0, a.shape[1]-1), axis=1)
    inside = (left>=0) & (right<a.shape[1])
    weight = (tick-left)/np.maximum(right-left, 1)
    initial = np.where(inside, pv+(nv-pv)*weight, historical)
    return initial, np.where(left>=0, (tick-left)/12, 24), np.where(right<a.shape[1], (right-tick)/12, 24)


class MemoryDayStore:
    def __init__(self, args):
        self.args = args
        log('Initializing dataset and road metadata...')
        self.exp = Experiment(args)
        e = self.exp
        profile = np.load(args.history)
        if list(profile['stations']) != e.stations:
            raise ValueError('Historical-profile station order mismatch')
        # Verify that the reused historical artifact precedes supervised dates.
        manifest = json.loads((Path(args.history).parent/'experiment.json').read_text())
        if manifest['historical_end'] >= args.train_start or manifest['panel'] != args.panel:
            raise ValueError('Historical profile overlaps training or belongs to another panel')
        self.history = profile['mean']
        e.history = self.history
        log('Reading Task 1 target template...')
        e.targets()
        log(f'Template ready: {len(e.day_targets)} dates')
        self.days = {}
        topology = pd.read_csv(e.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
        self.order = np.argsort([topology.loc[link, 'order_index'] for link in e.links])
        # Directed one/two-hop graph neighborhoods; key-zero is always self.
        adj = np.eye(e.n, dtype=bool)
        for groups in e.neighbors.values():
            for i, ids in enumerate(groups):
                adj[i, ids] = True
        adj = adj | adj.T
        hop2 = adj | ((adj.astype(int)@adj.astype(int))>0)
        self.graph = hop2
        self.direction = np.sign(np.array([topology.loc[l,'order_index'] for l in e.links])[None, :]
                                 - np.array([topology.loc[l,'order_index'] for l in e.links])[:, None])
        self.up = np.array([ids[0] if len(ids)==1 else -1 for ids in e.neighbors['up']])
        self.static = e.static[['lanes','length_km','free_speed_kmh','capacity_vph','critical_density','k_jam']].to_numpy(dtype='float32')
        self.ramp_files = {p.stem[-10:].replace('_','-'):p for p in (e.panel/'train/ramp_states').rglob('*.parquet')}
        self.ramp_map = pd.read_csv(e.panel/'network/ramp_attachment_map.csv')
        self.input_names = []
        for date, target in e.day_targets.items():
            if args.train_start <= date < args.val_end:
                log(f'Building features: {date}, {len(target):,} targets...')
                self.days[date] = self.build_day(date, target)
                log(f'Features ready: {date} ({len(self.days)} days loaded)')
        self.train = [d for date,d in self.days.items() if args.train_start<=date<args.val_start]
        self.valid = [d for date,d in self.days.items() if args.val_start<=date<args.val_end]
        if not self.train or not self.valid:
            raise ValueError('Empty train or holdout')

        # Fit feature scaling on supervised TRAINING dates only, before holdout.
        names = list(e.feature_grid(self.train[0]['date'])[0])
        if len(names) != 51:
            raise ValueError(f'Expected 51 base features, got {len(names)}')
        log('Fitting feature normalization on training dates...')
        sums = np.zeros(51); squares = np.zeros(51); counts = np.zeros(51)
        for day in self.train:
            if day['date'] >= args.internal_val_start:
                continue
            raw = day['features'].astype('float64')
            ok = np.isfinite(raw)
            sums += np.where(ok, raw, 0).sum(axis=(0,1))
            squares += np.where(ok, raw*raw, 0).sum(axis=(0,1))
            counts += ok.sum(axis=(0,1))
        self.mean = sums/np.maximum(counts,1)
        self.scale = np.sqrt(np.maximum(squares/np.maximum(counts,1)-self.mean**2, 0))
        self.scale[self.scale < 1e-6] = 1
        for day in self.days.values():
            raw = day['features']; ok = np.isfinite(raw)
            normalized = np.where(ok,(raw-self.mean)/self.scale,0)
            day['features'] = np.concatenate([normalized,ok],axis=-1).astype('float32')
        self.input_names = names + [name+'__finite' for name in names]
        self.feature_names = names
        log(f'Data ready: {len(self.days)} days, {len(self.input_names)} input channels')

    def ramps(self, date):
        e = self.exp
        net = np.zeros((e.n,288), dtype='float32')
        complete = np.ones_like(net, dtype=bool)
        if date not in self.ramp_files:
            return net, np.zeros_like(complete)
        r = pd.read_parquet(self.ramp_files[date])
        ts = pd.to_datetime(r.timestamp, utc=True)
        r = r.assign(slot=ts.dt.hour*12+ts.dt.minute//5)
        link_index = {link:i for i,link in enumerate(e.links)}
        grouped = {rid:g for rid,g in r.groupby('ramp_link_id')}
        for row in self.ramp_map.itertuples():
            if row.nearest_mainline_link_id not in link_index:
                continue
            i = link_index[row.nearest_mainline_link_id]
            g = grouped.get(row.ramp_link_id)
            if g is None:
                complete[i] = False
                continue
            q = np.full(288,np.nan)
            good = g.flow_vph.notna() & (g.pct_observed>=75)
            q[g.loc[good,'slot'].to_numpy(dtype=int)] = g.loc[good,'flow_vph']
            complete[i] &= np.isfinite(q)
            sign = 1 if str(row.ramp_type).upper() in ['OR','ON'] else -1
            net[i] += sign*np.nan_to_num(q)
        # Partial sums are hidden rather than interpreting missing ramps as zero.
        net[~complete] = 0
        return net,complete

    def build_day(self, date, target):
        e = self.exp
        x = pd.read_parquet(e.masked_files[date])
        wd = pd.Timestamp(date).dayofweek
        history = self.history[:,:,wd]
        v,q = [e.grid(x,col) for col in CHANNELS]
        q = q/e.lanes[:,None]
        bv,left,right = interpolate(v,history[0])
        bq,_,_ = interpolate(q,history[1])
        base = np.stack([bv/25,bq/600],axis=-1)
        visible = np.stack([np.isfinite(v),np.isfinite(q)],axis=-1)
        observed = np.stack([np.nan_to_num(v)/25,np.nan_to_num(q)/600],axis=-1)
        slot = np.broadcast_to(np.arange(288)[None,:],v.shape)
        occ = e.grid(x,'occupancy')
        net,complete = self.ramps(date) if self.args.conservation_weight else (np.zeros_like(v),np.zeros_like(v,dtype=bool))
        grids, bases, _, _ = e.feature_grid(date)
        features = np.stack(list(grids.values()),axis=-1)
        base = np.stack([bases[0]/25,bases[1]/600],axis=-1)
        target_mask = np.zeros(v.shape,dtype=bool)
        ts = pd.to_datetime(target.timestamp,utc=True)
        ii = target.station_id.map(e.si).to_numpy(dtype=int)
        jj = (ts.dt.hour*12+ts.dt.minute//5).to_numpy()
        if visible[ii,jj].any():
            raise ValueError('Hidden targets visible in inputs')
        old, _, _, _, _, names = e.features(date,target)
        np.testing.assert_allclose(features[ii,jj].astype('float32'),old,equal_nan=True,rtol=0,atol=0)
        target_mask[ii,jj] = True
        # Read labels last. They cannot affect features, initial values or graph.
        truth = pd.read_parquet(e.truth_files[date])
        y = np.stack([e.grid(truth,'speed_kmh')/25,
                      e.grid(truth,'flow_vph')/e.lanes[:,None]/600],axis=-1)
        if not np.isfinite(y[ii,jj]).all():
            raise ValueError('Nonfinite supervised label')
        result = dict(date=date,features=features,base=base,observed=observed,
                      visible=visible,label=np.nan_to_num(y),target=target_mask,
                      ramp=net,ramp_ok=complete,keys=target,ii=ii,jj=jj)
        for key in ['features','base','observed','visible','label','target','ramp','ramp_ok']:
            # [time, nodes, channels] for the attention model.
            result[key] = np.swapaxes(result[key],0,1).astype('float32')
        return result

    def graph_tensors(self, nodes):
        nodes = np.array(nodes)
        sub = self.graph[np.ix_(nodes,nodes)]
        max_neighbors = int(sub.sum(axis=1).max())
        index = np.zeros((len(nodes),max_neighbors),dtype='int64')
        valid = np.zeros_like(index,dtype=bool)
        direction = np.zeros_like(index)
        for i in range(len(nodes)):
            ids = np.flatnonzero(sub[i])
            index[i,:len(ids)] = ids
            valid[i,:len(ids)] = True
            direction[i,:len(ids)] = self.direction[nodes[i],nodes[ids]].astype(int)+1
        local = {int(v):i for i,v in enumerate(nodes)}
        upstream = np.array([local.get(int(self.up[v]),-1) for v in nodes])
        return dict(neighbors=torch.from_numpy(index),neighbor_valid=torch.from_numpy(valid),
                    direction=torch.from_numpy(direction),upstream=torch.from_numpy(upstream),
                    static=torch.from_numpy(self.static[nodes]))

    def batch(self, rng):
        n = min(self.args.patch_nodes,self.exp.n)
        start = int(rng.integers(self.exp.n-n+1))
        nodes = self.order[start:start+n]
        graph = self.graph_tensors(nodes)
        parts = []
        for _ in range(self.args.batch_size):
            day = self.train[int(rng.integers(len(self.train)))]
            t = int(rng.integers(289-self.args.window))
            parts.append({k:torch.from_numpy(day[k][t:t+self.args.window,nodes].copy())
                          for k in ['features','base','observed','visible','label','target','ramp','ramp_ok']})
        return {k:torch.stack([p[k] for p in parts]) for k in parts[0]},graph


class NormalizedFeatures:
    def __init__(self,path,mean,scale):
        self.path,self.mean,self.scale = path,mean,scale
    def __getitem__(self,index):
        raw=np.load(self.path,mmap_mode='r')[index]
        finite=np.isfinite(raw)
        return np.concatenate([np.where(finite,(raw-self.mean)/self.scale,0),finite],axis=-1).astype('float32')

class DiskDay:
    def __init__(self,date,folder,mean,scale):
        self.date,self.folder,self.mean,self.scale=date,folder,mean,scale
    def __getitem__(self,key):
        if key=='date': return self.date
        if key=='keys': return pd.read_parquet(self.folder/'keys.parquet')
        if key=='features': return NormalizedFeatures(self.folder/'features.npy',self.mean,self.scale)
        return np.load(self.folder/(key+'.npy'),mmap_mode='r')

class DayStore(MemoryDayStore):
    def __init__(self,args):
        self.args=args
        log('Initializing dataset and road metadata...')
        self.exp = Experiment(args)
        e = self.exp
        e.profile()
        (Path(args.history).parent/'experiment.json').write_text(json.dumps(dict(panel=args.panel,historical_end=e.profile_dates[-1],historical_start=e.profile_dates[0]),indent=2))
        profile = np.load(args.history)
        if list(profile['stations']) != e.stations:
            raise ValueError('Historical-profile station order mismatch')
        # Verify that the reused historical artifact precedes supervised dates.
        manifest = json.loads((Path(args.history).parent/'experiment.json').read_text())
        if manifest['historical_end'] >= args.train_start or manifest['panel'] != args.panel:
            raise ValueError('Historical profile overlaps training or belongs to another panel')
        self.history = profile['mean']
        e.history = self.history

        self.days = {}
        topology = pd.read_csv(e.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
        self.order = np.argsort([topology.loc[link, 'order_index'] for link in e.links])
        # Directed one/two-hop graph neighborhoods; key-zero is always self.
        adj = np.eye(e.n, dtype=bool)
        for groups in e.neighbors.values():
            for i, ids in enumerate(groups):
                adj[i, ids] = True
        adj = adj | adj.T
        hop2 = adj | ((adj.astype(int)@adj.astype(int))>0)
        self.graph = hop2
        self.direction = np.sign(np.array([topology.loc[l,'order_index'] for l in e.links])[None, :]
                                 - np.array([topology.loc[l,'order_index'] for l in e.links])[:, None])
        self.up = np.array([ids[0] if len(ids)==1 else -1 for ids in e.neighbors['up']])
        self.static = e.static[['lanes','length_km','free_speed_kmh','capacity_vph','critical_density','k_jam']].to_numpy(dtype='float32')
        self.ramp_files = {p.stem[-10:].replace('_','-'):p for p in (e.panel/'train/ramp_states').rglob('*.parquet')}
        self.ramp_map = pd.read_csv(e.panel/'network/ramp_attachment_map.csv')
        cache=Path(args.output)/'cache'
        cache.mkdir(exist_ok=True)
        marker=cache/'ready.json'
        import hashlib
        sources=[Path(__file__),Path(__file__).with_name('task1_baseline.py')]
        signature=dict(version='shared51-disk-v2',panel=args.panel,train_start=args.train_start,
                       internal_val_start=args.internal_val_start,val_start=args.val_start,val_end=args.val_end,
                       code_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in sources])
        source_files=list(e.masked_files.values())+list(e.truth_files.values())+list((e.panel/'network').glob('*.csv'))
        source_files.append(e.root/'task1'/args.panel/'train/sample_submission_state.csv')
        signature['data_files']=[dict(path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(source_files)]
        if marker.exists():
            meta=json.loads(marker.read_text())
            if meta['signature']!=signature: raise ValueError('Cache signature mismatch; choose a new experiment output')
            names,mean,scale,dates=meta['names'],np.array(meta['mean']),np.array(meta['scale']),meta['dates']
            log(f'Reusing {len(dates)} cached days')
        else:
            # Partition template by date without retaining all target keys in RAM.
            parts=Path(tempfile.mkdtemp(prefix='template_',dir=cache))
            grouped={}
            path=e.root/'task1'/args.panel/'train/sample_submission_state.csv'
            log('Partitioning target template by date...')
            for chunk_id,chunk in enumerate(pd.read_csv(path,usecols=KEYS,dtype=str,chunksize=250000)):
                ds=chunk.timestamp.str[:10]
                chunk=chunk.loc[(ds>=args.train_start)&(ds<args.val_end)]
                for date,g in chunk.groupby(chunk.timestamp.str[:10],sort=False):
                    p=parts/f'{date}_{chunk_id}.parquet'; g.to_parquet(p,index=False)
                    grouped.setdefault(date,[]).append(p)
                if chunk_id%10==0: log(f'Template chunks read: {chunk_id+1}')
            dates=sorted(grouped)
            sums=np.zeros(51); squares=np.zeros(51); counts=np.zeros(51)
            for date in dates:
                log(f'Caching {args.panel}: {date} ({dates.index(date)+1}/{len(dates)})')
                target=pd.concat([pd.read_parquet(p) for p in grouped[date]],ignore_index=True)
                if target.duplicated(KEYS).any(): raise ValueError('Duplicate template keys')
                day=self.build_day(date,target)
                raw=day['features']; names=list(e.feature_grid(date)[0])
                if len(names)!=51: raise ValueError('Expected 51 features')
                if date<args.internal_val_start:
                    a=raw.astype('float64'); good=np.isfinite(a)
                    sums+=np.where(good,a,0).sum(axis=(0,1))
                    squares+=np.where(good,a*a,0).sum(axis=(0,1)); counts+=good.sum(axis=(0,1))
                folder=cache/date; folder.mkdir(exist_ok=True)
                for key in ['features','base','observed','visible','label','target','ramp','ramp_ok','ii','jj']:
                    np.save(folder/(key+'.npy'),day[key],allow_pickle=False)
                target.to_parquet(folder/'keys.parquet',index=False)
                for p in grouped[date]: p.unlink()
                del day,raw,target
            parts.rmdir()
            mean=sums/np.maximum(counts,1)
            scale=np.sqrt(np.maximum(squares/np.maximum(counts,1)-mean**2,0)); scale[scale<1e-6]=1
            meta=dict(signature=signature,names=names,mean=mean.tolist(),scale=scale.tolist(),dates=dates)
            marker.write_text(json.dumps(meta,indent=2))
        self.mean,self.scale=mean,scale
        self.feature_names=names; self.input_names=names+[name+'__finite' for name in names]
        self.days={date:DiskDay(date,cache/date,mean,scale) for date in dates}
        self.train=[d for date,d in self.days.items() if args.train_start<=date<args.val_start]
        self.valid=[d for date,d in self.days.items() if args.val_start<=date<args.val_end]
        expected=sorted(date for date in e.masked_files if args.train_start<=date<args.val_end)
        if dates!=expected: raise ValueError('Some dates have no template targets')
        if not self.train or not self.valid: raise ValueError('Empty split')
        log(f'Disk dataset ready: {len(dates)} days, 102 channels, {e.n} nodes')

class GraphTimeBlock(nn.Module):
    def __init__(self,width,heads,dropout):
        super().__init__()
        self.heads,self.d = heads,width//heads
        self.tnorm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.snorm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width,3*width)
        self.direction_bias = nn.Embedding(3,heads)
        self.project = nn.Linear(width,width)
        self.ff = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4*width),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(4*width,width))
        self.drop = nn.Dropout(dropout)

    def forward(self,x,g):
        b,t,n,w = x.shape
        z = self.tnorm(x).permute(0,2,1,3).reshape(b*n,t,w)
        z = self.temporal(z,z,z,need_weights=False)[0]
        x = x+self.drop(z.reshape(b,n,t,w).permute(0,2,1,3))
        q,k,v = self.qkv(self.snorm(x)).chunk(3,dim=-1)
        q = q.reshape(b,t,n,self.heads,self.d)
        k = k.reshape(b,t,n,self.heads,self.d)[:,:,g['neighbors']]
        v = v.reshape(b,t,n,self.heads,self.d)[:,:,g['neighbors']]
        logits = (q[:,:,:,None]*k).sum(-1)/math.sqrt(self.d)
        logits = logits+self.direction_bias(g['direction'])[None,None]
        logits = logits.masked_fill(~g['neighbor_valid'][None,None,:,:,None],-1e4)
        attention = torch.softmax(logits,dim=3)
        message = (attention[...,None]*v).sum(dim=3).reshape(b,t,n,w)
        x = x+self.drop(self.project(message))
        return x+self.drop(self.ff(x))


class StateTransformer(nn.Module):
    def __init__(self,features,width,heads,layers,window,dropout):
        super().__init__()
        self.input = nn.Linear(features,width)
        self.position = nn.Parameter(torch.randn(1,window,1,width)*.01)
        self.blocks = nn.ModuleList([GraphTimeBlock(width,heads,dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('road_scale',torch.tensor([6.,2.,120.,12000.,120.,600.]))

    def forward(self,batch,g):
        x = self.input(batch['features'])
        x = x+self.position[:,:x.shape[1]]
        for block in self.blocks:
            x = block(x,g)
        pred = batch['base']+self.head(x)
        # Both quantities are constrained positive. Density is derived, not a
        # separately adjustable output which could artificially satisfy physics.
        return torch.stack([pred[...,0].clamp_min(.04),pred[...,1].clamp_min(0)],dim=-1)


def objective(pred,b,g,args):
    mask = b['target'].bool()
    if not mask.any():
        return pred.sum()*0,dict(reconstruction=0.,fd=0.,conservation_proxy=0.)
    err = pred[mask]-b['label'][mask]
    rmse = torch.sqrt((err**2).mean(dim=0)+1e-8)
    reconstruction = .54*rmse[0]+.46*rmse[1]
    if args.fd_weight == 0 and args.conservation_weight == 0:
        return reconstruction, dict(reconstruction=float(reconstruction.detach()),fd=0.,conservation_proxy=0.)
    state = torch.where(b['visible'].bool(),b['observed'],pred)
    lanes,length,vf,capacity,kcrit,kjam = g['static'].unbind(-1)
    v = state[...,0]*25
    q = state[...,1]*600*lanes
    k = q/v.clamp_min(1)
    wave = capacity/(kjam-kcrit).clamp_min(1)
    envelope = torch.minimum(vf*k,wave*(kjam-k)).clamp_min(0)
    # Five-percent capacity tolerance avoids fitting measurement noise exactly.
    violation = F.relu((q-envelope)/capacity-.05)
    fd = (violation[mask]**2).mean()
    up = g['upstream']
    flux_in = q[:,:,up.clamp_min(0)]
    residual = (k[:,1:]-k[:,:-1])*length-(flux_in[:,:-1]+b['ramp'][:,:-1]-q[:,:-1])/12
    valid = b['ramp_ok'][:,:-1].bool() & (up>=0)[None,None]
    # At least one endpoint is a target. Observed values are held fixed.
    valid &= mask[:,1:]|mask[:,:-1]
    normalized = residual/(capacity/12).clamp_min(1)
    conservation = F.smooth_l1_loss(normalized[valid],torch.zeros_like(normalized[valid])) if valid.any() else pred.sum()*0
    loss = reconstruction+args.fd_weight*fd+args.conservation_weight*conservation
    return loss,dict(reconstruction=float(reconstruction.detach()),fd=float(fd.detach()),
                     conservation_proxy=float(conservation.detach()))


@torch.no_grad()
def evaluate(model,store,args):
    model.eval()
    g = {k:v.to(args.device) for k,v in store.graph_tensors(np.arange(store.exp.n)).items()}
    truths,predictions,regimes,keys = [],[],[],[]
    window = args.window
    # Fifty-percent overlap; predictions are averaged before evaluating targets.
    starts = list(range(0,289-window,max(1,window//2)))
    if starts[-1] != 288-window:
        starts.append(288-window)
    for day in store.valid:
        log(f'Evaluating {day["date"]}...')
        total = np.zeros((288,store.exp.n,2),dtype='float64')
        weight = np.zeros((288,1,1))
        for t in starts:
            batch = {k:torch.from_numpy(day[k][None,t:t+window].copy()).to(args.device) for k in ['features','base']}
            total[t:t+window] += model(batch,g)[0].cpu().numpy()
            weight[t:t+window] += 1
        reconstruction = total/weight
        ii,jj = day['ii'],day['jj']
        scale = np.array([25.,600.])
        predictions.append(reconstruction[jj,ii]*scale)
        truths.append(day['label'][jj,ii]*scale)
        regimes.append(day['keys'].mask_regime.to_numpy())
        keys.append(day['keys'])
    pred = np.concatenate(predictions)
    rows = score(np.concatenate(truths),pred,np.concatenate(regimes),'graph_transformer')
    output = pd.concat(keys,ignore_index=True)
    output['speed_kmh'] = pred[:,0]
    lanes = output.station_id.map(dict(zip(store.exp.stations,store.exp.lanes))).to_numpy()
    output['flow_vph'] = pred[:,1]*lanes
    return rows,output


def run(args):
    started = time.time()
    if (Path(args.output)/'last.pt').exists() and not args.resume:
        raise ValueError('Existing checkpoint: use --resume or a new output')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    log(f'Starting; device={args.device}, data={args.release_root}')
    store = DayStore(args)
    if args.prepare_only:
        b,g=store.batch(rng)
        b={k:v.to(args.device) for k,v in b.items()}; g={k:v.to(args.device) for k,v in g.items()}
        test=StateTransformer(102,args.width,args.heads,args.layers,args.window,args.dropout).to(args.device)
        pred=test(b,g)
        torch.testing.assert_close(pred,b['base'].clamp_min(0),rtol=0,atol=.04)
        loss,_=objective(pred,b,g,args); loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in test.parameters() if p.grad is not None):
            raise ValueError('Nonfinite smoke gradient')
        log('PREPARE CHECK PASSED: disk slices, CUDA forward/backward, finite gradients')
        return
    internal = [d for d in store.train if d['date'] >= args.internal_val_start]
    store.train = [d for d in store.train if d['date'] < args.internal_val_start]
    if not internal or not store.train:
        raise ValueError('Internal validation must split the supervised training block')
    model = StateTransformer(len(store.input_names),args.width,args.heads,args.layers,args.window,args.dropout).to(args.device)
    parameter_count = sum(p.numel() for p in model.parameters())
    log(f'Model: {parameter_count:,} parameters, {len(store.train)} training days, {len(store.valid)} holdout days, device={args.device}')
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs*args.steps_per_epoch,eta_min=args.learning_rate*.1)
    output = Path(args.output)
    if (output/'last.pt').exists() and not args.resume:
        raise ValueError('Output has an existing experiment; use a new output or --resume')
    output.mkdir(parents=True,exist_ok=True)
    (output/'run_config.json').write_text(json.dumps(vars(args),indent=2))
    best = -float('inf'); stale = 0; first_epoch = 0; trace = []
    if args.resume:
        ck = torch.load(args.resume,map_location=args.device,weights_only=False)
        for key in ['train_start','val_start','val_end','internal_val_start','seed','epochs','steps_per_epoch','panel','window','patch_nodes','batch_size','width','heads','layers','dropout','learning_rate','fd_weight','conservation_weight','history','release_root']:
            if ck['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(ck['state_dict']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); first_epoch=ck['epoch']
        best=ck['best']; stale=ck['stale']; trace=ck['trace']
        rng.bit_generator.state=ck['numpy_rng']; torch.set_rng_state(ck['torch_rng'].cpu())
        if torch.cuda.is_available() and ck['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in ck['cuda_rng']])
    for epoch in range(first_epoch,args.epochs):
        model.train()
        records = []
        for step in range(args.steps_per_epoch):
            b,g = store.batch(rng)
            b = {k:v.to(args.device) for k,v in b.items()}
            g = {k:v.to(args.device) for k,v in g.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(b,g)
            loss,terms = objective(pred,b,g,args)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            records.append(dict(loss=float(loss.detach()),**terms))
            if step == 0 or (step+1)%32==0 or step+1 == args.steps_per_epoch:
                log(f'Epoch {epoch+1}/{args.epochs}, step {step+1}/{args.steps_per_epoch}, loss={np.mean([r["loss"] for r in records]):.6f}')
        trace.append(dict(epoch=epoch+1,**pd.DataFrame(records).mean().to_dict()))
        final_days = store.valid
        store.valid = internal
        internal_rows, _ = evaluate(model,store,args)
        store.valid = final_days
        value = internal_rows[-1]['S_state']
        trace[-1]['internal_S_state'] = value
        improved = value > best
        stale = 0 if improved else stale+1
        best = max(best,value)
        ck = dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                  scheduler=scheduler.state_dict(),epoch=epoch+1,best=best,stale=stale,
                  trace=trace,args=vars(args),input_names=store.input_names,
                  feature_mean=store.mean,feature_scale=store.scale,
                  stations=store.exp.stations,links=store.exp.links,
                  numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.save(ck,output/'last.tmp')
        (output/'last.tmp').replace(output/'last.pt')
        if improved:
            torch.save(ck,output/'best.pt')
        pd.DataFrame(trace).to_csv(output/'training_trace.csv',index=False)
        log(f'Internal validation S_state={value:.7f}, best={best:.7f}')
        if stale >= args.patience:
            break
    model.load_state_dict(torch.load(output/'best.pt',map_location=args.device,weights_only=False)['state_dict'])
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),args=vars(args),input_names=store.input_names,
                    stations=store.exp.stations,links=store.exp.links,feature_mean=store.mean,feature_scale=store.scale),output/'model.pt')
    log('Training finished. Evaluating reserved chronological holdout...')
    rows,predictions = evaluate(model,store,args)
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    predictions.to_parquet(output/'holdout_predictions.parquet',index=False)
    pd.DataFrame(trace).to_csv(output/'training_trace.csv',index=False)
    report = dict(vars(args),parameter_count=parameter_count,input_features=store.input_names,
                  train_days=len(store.train),holdout_days=len(store.valid),holdout_targets=len(predictions),
                  elapsed_seconds=time.time()-started,
                  limitations=['Custom architecture; not a SAITS/PDFormer reproduction.',
                               'Conservation is a topology-derived detector-flow proxy, not exact boundary-flux physics.',
                               'The interpolation baseline is an explicit residual skip connection.',
                               'Checkpoint selected on internal validation; last 20 percent dates used only for final reporting.',
                               'One corridor per model, chronological local holdout; not official competition scores.'])
    (output/'experiment.json').write_text(json.dumps(report,indent=2)+'\n')
    log(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--release-root',required=True)
    p.add_argument('--internal-val-start',default='2031-01-25')
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--resume',default=None)
    p.add_argument('--panel',default='D12_I5_N')
    p.add_argument('--history',default=str(Path(__file__).parent/'task1_pilot/history.npz'))
    p.add_argument('--train-start',default='2031-01-01')
    p.add_argument('--val-start',default='2031-02-01')
    p.add_argument('--val-end',default='2031-03-01')
    p.add_argument('--output',default=str(Path(__file__).parent/'task1_graph_transformer'))
    p.add_argument('--window',type=int,default=24)
    p.add_argument('--patch-nodes',type=int,default=64)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--steps-per-epoch',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=.0003)
    p.add_argument('--fd-weight',type=float,default=0.)
    p.add_argument('--conservation-weight',type=float,default=0.)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda','mps'])
    a = p.parse_args()
    if a.width%a.heads or not a.train_start<a.val_start<a.val_end or a.window<2 or a.window>288:
        p.error('Invalid width/heads, split dates or window')
    if not a.train_start < a.internal_val_start < a.val_start or min(a.epochs,a.steps_per_epoch,a.batch_size,a.patch_nodes,a.patience) < 1:
        p.error('Invalid internal split or nonpositive training parameter')
    with RunLock(Path(a.output)/'worker.lock'):
        run(a)
