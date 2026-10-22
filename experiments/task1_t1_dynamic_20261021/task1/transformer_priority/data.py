from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from .common import HERE, KEY, SCALE, NeuralCache, read, snapshot, sha256, upstream


def graph_from(exp):
    topology=pd.read_csv(exp.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
    order_values=np.array([topology.loc[link,'order_index'] for link in exp.links])
    adj=np.eye(exp.n,dtype=bool)
    for groups in exp.neighbors.values():
        for i,ids in enumerate(groups):
            adj[i,ids]=True
    adj=adj|adj.T
    graph=adj|((adj.astype(np.int32)@adj.astype(np.int32))>0)
    return dict(graph=graph,direction=np.sign(order_values[None,:]-order_values[:,None]).astype(np.int64),
                order=np.argsort(order_values),isolated=graph.sum(1)==1)


def graph_tensors(graph,nodes):
    nodes=np.asarray(nodes,dtype=int)
    if len(np.unique(nodes))!=len(nodes):
        raise ValueError('Duplicate nodes in patch')
    sub=graph['graph'][np.ix_(nodes,nodes)]
    width=int(sub.sum(1).max())
    neighbors=np.zeros((len(nodes),width),dtype=np.int64)
    valid=np.zeros_like(neighbors,dtype=bool);direction=np.zeros_like(neighbors)
    for i,row in enumerate(sub):
        ids=np.flatnonzero(row);neighbors[i,:len(ids)]=ids;valid[i,:len(ids)]=True
        direction[i,:len(ids)]=graph['direction'][nodes[i],nodes[ids]]+1
    return dict(neighbors=torch.from_numpy(neighbors),neighbor_valid=torch.from_numpy(valid),direction=torch.from_numpy(direction))


def sampling_spec(rng,n,order,dates,window,patch,batch,balanced):
    width=min(n,patch)
    start=int(rng.integers(n if balanced else n-width+1))
    positions=(start+np.arange(width))%n if balanced else np.arange(start,start+width)
    nodes=order[positions]
    samples=[]
    for _ in range(batch):
        date=dates[int(rng.integers(len(dates)))]
        tick=int(rng.integers(-(window-1),288)) if balanced else int(rng.integers(289-window))
        samples.append((date,tick))
    return nodes,samples


def reliability_from(raw,names,visible):
    result=[]
    for c,channel in enumerate(['speed_kmh','flow_vph']):
        before=raw[...,names.index(channel+'_prev_gap_min')]
        after=raw[...,names.index(channel+'_next_gap_min')]
        result.extend([visible[...,c].astype(np.float32),np.nan_to_num(before/60,nan=24.).clip(0,24),
                       np.nan_to_num(after/60,nan=24.).clip(0,24),(np.isfinite(before)&np.isfinite(after)).astype(np.float32)])
    return np.stack(result,axis=-1).astype(np.float32)


def prepare(root,panel,cache_root,plan,smoke=False):
    cache=NeuralCache(cache_root)
    feature_sources=snapshot([HERE/'data.py',HERE/'common.py',HERE/'upstream/task1_baseline.py'])
    network=snapshot((root/'corridors'/panel/'network').glob('*.csv'))
    template=root/'task1'/panel/'train/sample_submission_state.csv'
    all_dates=plan['training_dates']+plan['internal_validation_dates']+plan['holdout_dates']
    identity=dict(kind='tt_template_dates',panel=panel,source=snapshot([template]),dates=all_dates,implementation=feature_sources,schema=1)
    def partition(out):
        parts={}
        scratch=out/'parts';scratch.mkdir()
        for k,chunk in enumerate(pd.read_csv(template,usecols=KEY,dtype=str,chunksize=250000)):
            dates=chunk.timestamp.str[:10];chunk=chunk.loc[dates.isin(all_dates)]
            for date,group in chunk.groupby(chunk.timestamp.str[:10],sort=False):
                path=scratch/(date+'_'+str(k)+'.parquet');group.to_parquet(path,index=False);parts.setdefault(date,[]).append(path)
        if sorted(parts)!=all_dates:
            raise ValueError('Template dates incomplete')
        records=[]
        for date in all_dates:
            keys=pd.concat([pd.read_parquet(p) for p in parts[date]],ignore_index=True)[KEY]
            if keys.isna().any().any() or keys.duplicated(KEY).any() or keys.duplicated(['timestamp','station_id']).any():
                raise ValueError('Null or duplicate template keys')
            keys.to_parquet(out/(date+'.parquet'),index=False)
            records.append(dict(date=date,rows=len(keys),regimes=keys.mask_regime.value_counts().to_dict()))
            for p in parts[date]:
                p.unlink()
        # An empty scratch directory is harmless; Windows cloud-folder attributes
        # can reject rmdir even after all known temporary files were removed.
        return dict(dates=records,target_rows=sum(r['rows'] for r in records))
    template_folder,template_manifest,template_hit=cache.get_or_build(identity,partition)
    metadata={r['date']:r for r in template_manifest['metadata']['dates']}
    def select(dates):
        found=[]
        for regime in ['R1','R2','R3']:
            match=next((d for d in dates if regime in metadata[d]['regimes']),None)
            if match is None:
                raise ValueError('Missing smoke regime '+regime)
            found.append(match)
        return sorted(set(found))
    roles=dict(training=plan['training_dates'][:14] if smoke else plan['training_dates'],
               internal=select(plan['internal_validation_dates']) if smoke else plan['internal_validation_dates'],
               holdout=select(plan['holdout_dates']) if smoke else plan['holdout_dates'])
    args=SimpleNamespace(release_root=str(root),panel=panel,output=str(cache_root/'metadata_probe'),seed=2026,
                         train_start=plan['training_dates'][0],val_start=plan['holdout_dates'][0],val_end='2031-03-01')
    # The probe directory is part of this new cache only; original outputs are never used.
    exp=upstream().baseline.Experiment(args)
    if sorted(exp.masked_files)!=plan['history_dates']+all_dates or sorted(exp.truth_files)!=plan['history_dates']+all_dates:
        raise ValueError('Released chronological dates differ')
    history_inputs=snapshot([exp.truth_files[d] for d in plan['history_dates']])
    history_identity=dict(kind='tt_history28',panel=panel,inputs=history_inputs,network=network,implementation=feature_sources,dates=plan['history_dates'])
    def history_builder(out):
        old=exp.out;exp.out=out
        try:
            exp.profile()
        finally:
            exp.out=old
        if exp.profile_dates!=plan['history_dates']:
            raise ValueError('Historical date overlap')
        return dict(stations=exp.stations,links=exp.links,dates=exp.profile_dates)
    history_folder,history_manifest,history_hit=cache.get_or_build(history_identity,history_builder)
    with np.load(history_folder/'history.npz',allow_pickle=False) as archive:
        if list(archive['stations'])!=exp.stations or list(archive['links'])!=exp.links:
            raise ValueError('History node order differs')
        exp.history=archive['mean'].copy()
    expected_names=read(HERE/'reference_feature_names.json')
    records={};inputs=network+history_inputs+snapshot([template])
    for date in roles['training']+roles['internal']+roles['holdout']:
        source=snapshot([exp.masked_files[date]]);inputs+=source
        identity=dict(kind='tt_masked_day51',panel=panel,date=date,masked=source,network=network,
                      implementation=feature_sources,history=history_manifest['key'],template=template_manifest['key'])
        def build_day(out,date=date):
            keys=pd.read_parquet(template_folder/(date+'.parquet'))
            features,bases,_,masked=exp.feature_grid(date)
            names=list(features)
            if names!=expected_names or len(names)!=51:
                raise ValueError('51-feature order changed')
            raw=np.swapaxes(np.stack(list(features.values()),axis=-1),0,1).astype(np.float32)
            base=np.swapaxes(np.stack([bases[0]/25,bases[1]/600],axis=-1),0,1).astype(np.float32)
            visible=np.swapaxes(np.stack([np.isfinite(exp.grid(masked,c)) for c in ['speed_kmh','flow_vph']],axis=-1),0,1)
            ii=keys.station_id.map(exp.si).to_numpy(int)
            ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
            if visible[jj,ii].any() or not np.array_equal(keys.link_id.to_numpy(),np.array(exp.links)[ii]):
                raise ValueError('Target visibility or node alignment failed')
            target=np.zeros((288,exp.n),dtype=bool);target[jj,ii]=True
            if target.sum()!=len(keys) or np.isinf(raw).any() or not np.isfinite(base).all():
                raise ValueError('Invalid feature cache')
            for name,value in dict(features=raw,base=base,visible=visible,target=target,ii=ii,jj=jj,
                                   reliability=reliability_from(raw,names,visible)).items():
                np.save(out/(name+'.npy'),value,allow_pickle=False)
            keys.to_parquet(out/'keys.parquet',index=False)
            return dict(rows=len(keys),date=date,nodes=exp.n,feature_names=names)
        folder,manifest,hit=cache.get_or_build(identity,build_day)
        records[date]=dict(folder=str(folder),key=manifest['key'],hit=hit,rows=manifest['metadata']['rows'])
    identity=dict(kind='tt_training_normalization51',panel=panel,dates=roles['training'],
                  days=[records[d]['key'] for d in roles['training']],implementation=feature_sources)
    def normalization(out):
        sums=np.zeros(51);squares=np.zeros(51);counts=np.zeros(51)
        for date in roles['training']:
            raw=np.load(Path(records[date]['folder'])/'features.npy',mmap_mode='r').astype(np.float64)
            finite=np.isfinite(raw)
            sums+=np.where(finite,raw,0).sum((0,1));squares+=np.where(finite,raw*raw,0).sum((0,1));counts+=finite.sum((0,1))
        mean=sums/np.maximum(counts,1);scale=np.sqrt(np.maximum(squares/np.maximum(counts,1)-mean**2,0));scale[scale<1e-6]=1
        np.save(out/'mean.npy',mean);np.save(out/'scale.npy',scale)
        return dict(fit_dates=roles['training'],training_only=True)
    norm_folder,norm,norm_hit=cache.get_or_build(identity,normalization)
    return CachedData(exp,cache,records,roles,graph_from(exp),np.load(norm_folder/'mean.npy'),np.load(norm_folder/'scale.npy'),
                      dict(template_key=template_manifest['key'],history_key=history_manifest['key'],normalization_key=norm['key'],
                           feature_names=expected_names,days=records,roles=roles,inputs=inputs,history_hit=history_hit,
                           template_hit=template_hit,normalization_hit=norm_hit))


class CachedData:
    def __init__(self,exp,cache,records,roles,graph,mean,scale,provenance):
        self.exp,self.cache,self.records,self.roles,self.graph=exp,cache,records,roles,graph
        self.mean,self.scale,self.provenance=mean,scale,provenance
        self.names=provenance['feature_names'];self.n=exp.n;self.panel=exp.args.panel;self.family=self.panel.rsplit('_',1)[0]
        self.allow_holdout=False;self.label_records={};self._labels={};self._grids=OrderedDict()

    @lru_cache(maxsize=128)
    def array(self,date,name):
        return np.load(Path(self.records[date]['folder'])/(name+'.npy'),mmap_mode='r',allow_pickle=False)

    @lru_cache(maxsize=245)
    def keys(self,date):
        return pd.read_parquet(Path(self.records[date]['folder'])/'keys.parquet')

    def labels(self,date,purpose):
        if purpose not in self.roles or date not in self.roles[purpose] or (purpose=='holdout' and not self.allow_holdout):
            raise ValueError('Label access outside current phase: '+date+' / '+purpose)
        if date not in self._labels:
            identity=dict(kind='tt_compact_truth',panel=self.panel,date=date,targets=self.records[date]['key'],
                          truth=snapshot([self.exp.truth_files[date]]),schema=1)
            def build(out):
                truth=pd.read_parquet(self.exp.truth_files[date]);ii=self.array(date,'ii');jj=self.array(date,'jj')
                y=np.column_stack([self.exp.grid(truth,c)[ii,jj] for c in ['speed_kmh','flow_vph']])
                y[:,1]/=self.exp.lanes[ii]
                if not np.isfinite(y).all():
                    raise ValueError('Nonfinite template truth')
                np.save(out/'truth.npy',y.astype(np.float64))
                return dict(rows=len(y),purpose_scope='date-role guard is enforced by dataset')
            folder,manifest,hit=self.cache.get_or_build(identity,build)
            self._labels[date]=np.load(folder/'truth.npy',allow_pickle=False)
            self.label_records[date]=dict(folder=str(folder),key=manifest['key'],hit=hit,source=identity['truth'])
        return self._labels[date]

    def label_grid(self,date):
        if date not in self._grids:
            y=self.labels(date,'training');grid=np.zeros((288,self.n,2),dtype=np.float32)
            grid[self.array(date,'jj'),self.array(date,'ii')]=(y/SCALE).astype(np.float32)
            self._grids[date]=grid
            if len(self._grids)>12:
                self._grids.popitem(last=False)
        self._grids.move_to_end(date)
        return self._grids[date]

    def block(self,date,start,nodes,window,supervised):
        ticks=np.arange(start,start+window);valid=(ticks>=0)&(ticks<288);ticks=ticks.clip(0,287)
        raw=np.asarray(self.array(date,'features')[ticks][:,nodes]).copy()
        raw[~valid]=np.nan
        finite=np.isfinite(raw)
        features=np.concatenate([np.where(finite,(raw-self.mean)/self.scale,0),finite],axis=-1).astype(np.float32)
        result=dict(features=features,time_valid=valid)
        for name in ['base','reliability','visible']:
            value=np.asarray(self.array(date,name)[ticks][:,nodes]).copy();value[~valid]=0;result[name]=value
        if supervised:
            target=np.asarray(self.array(date,'target')[ticks][:,nodes]).copy();target[~valid]=False
            label=self.label_grid(date)[ticks][:,nodes].copy();label[~valid]=0
            result.update(target=target,label=label)
        return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in result.items()}

    def batch(self,rng,config,balanced):
        nodes,samples=sampling_spec(rng,self.n,self.graph['order'],self.roles['training'],config['window'],config['patch_nodes'],config['batch_size'],balanced)
        parts=[self.block(date,tick,nodes,config['window'],True) for date,tick in samples]
        return {k:torch.stack([p[k] for p in parts]) for k in parts[0]},graph_tensors(self.graph,nodes),dict(nodes=nodes.tolist(),samples=samples)

    def target_arrays(self,date):
        ii,jj=self.array(date,'ii'),self.array(date,'jj')
        return np.asarray(self.array(date,'features')[jj,ii]),np.asarray(self.array(date,'base')[jj,ii],dtype=np.float64)*SCALE

    def truth(self,dates,purpose):
        return np.concatenate([self.labels(d,purpose) for d in dates])

    def expected(self,dates):
        return pd.concat([self.keys(d) for d in dates],ignore_index=True)
