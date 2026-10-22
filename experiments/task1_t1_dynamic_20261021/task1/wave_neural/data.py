"""Read-only reuse of existing grids/waves, with separate input/ramp caches."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..cache import snapshot
from ..data import read_csv, sha256
from ..features import temporal_context
from ..grin.data import time_covariates, window_starts
from .physics import structure, ramp_grid


def implementation():
    folder = Path(__file__).parent
    return snapshot([folder/'data.py', folder/'physics.py', folder.parent/'features.py', folder.parent/'grin/data.py'])


def training_statistics(cache, folder, manifest):
    identity = dict(kind='wave_neural_statistics', schema='visible_station_mean_v1', panel=manifest['identity']['panel'],
                    training_grid_key=manifest['key'], implementation=implementation(), numpy=importlib.metadata.version('numpy'))
    def build(destination):
        n = len(manifest['metadata']['nodes'])
        sums, count = np.zeros((n,2),np.float64), np.zeros((n,2),np.int64)
        for date in manifest['metadata']['dates']:
            with np.load(folder/(date+'.npz'),allow_pickle=False) as a:
                x, mask = a['x'], a['mask']
            if not np.isfinite(x[mask]).all():
                raise ValueError('Nonfinite eligible training observation')
            sums += np.where(mask,x,0).sum(0,dtype=np.float64)
            count += mask.sum(0)
        global_mean = sums.sum(0)/count.sum(0)
        means = np.divide(sums,count,out=np.broadcast_to(global_mean,sums.shape).copy(),where=count>0)
        if not np.isfinite(means).all() or (means<0).any():
            raise ValueError('Invalid visible training means')
        np.save(destination/'means.npy',means.astype(np.float32))
        return dict(fit_dates=manifest['metadata']['dates'], fit_on='eligible visible training x/mask only; never y', nodes=n)
    return cache.get_or_build(identity,build)


def physical_cache(cache, root, panel, nodes):
    folder = root/'corridors'/panel/'network'
    names = ['fd_parameters.csv','lwr_mainline_topology.csv','ramp_attachment_map.csv']
    identity = dict(kind='wave_neural_physical_structure',schema='strict_midpoint_proxy_v1',panel=panel,
                    nodes=nodes.to_dict('records'),inputs=snapshot(folder/name for name in names),implementation=implementation())
    def build(destination):
        fd = read_csv(folder/'fd_parameters.csv').set_index('link_id')
        arrays,meta = structure(nodes,fd,read_csv(folder/'lwr_mainline_topology.csv'),read_csv(folder/'ramp_attachment_map.csv'))
        if not arrays['valid_nodes'].any():
            raise ValueError('No complete internal control volumes for the conservation ablation')
        np.savez_compressed(destination/'structure.npz',**arrays)
        par=arrays['parameters'];lanes=par[:,0]
        attrs=np.stack([par[:,2]/120,par[:,3]/lanes/2000,lanes/8,par[:,1]/2,par[:,4]/lanes/50,par[:,5]/lanes/150],-1)
        np.save(destination/'attributes.npy',attrs.astype(np.float32))
        return meta
    return cache.get_or_build(identity,build)


def ramp_files(root,panel):
    result={}
    for path in sorted((root/'corridors'/panel/'train/ramp_states').rglob('*.parquet')):
        date='-'.join(path.stem.split('_')[-3:])
        if len(date)!=10 or date in result:
            raise ValueError('Unrecognized or duplicate daily ramp file')
        result[date]=path
    return result


def input_day(cache,date,grid_folder,grid_manifest,wave_folder,wave_manifest,stats_folder,stats,physical,ramps,attachments,nodes):
    source=grid_folder/(date+'.npz')
    if date not in grid_manifest['metadata']['dates'] or wave_manifest['identity']['date']!=date:
        raise ValueError('Daily source date differs')
    wi=wave_manifest['identity']
    if wi['source_grid_key']!=grid_manifest['key'] or wi['source_archive_sha256']!=sha256(source) or wi['truth_used']:
        raise ValueError('Wave belongs to a different observation grid')
    identity=dict(kind='wave_neural_day_input',schema='temporal_ramp_input_v1',panel=grid_manifest['identity']['panel'],date=date,
                  grid_key=grid_manifest['key'],grid_archive_sha256=sha256(source),wave_key=wave_manifest['key'],
                  statistics_key=stats['key'],physical_structure_key=physical['key'],ramp_inputs=snapshot([ramps]),implementation=implementation(),
                  truth_used=False)
    def build(destination):
        with np.load(source,allow_pickle=False) as a:
            x,mask=a['x'],a['mask']
        means=np.load(stats_folder/'means.npy',allow_pickle=False)
        interp=np.empty_like(x)
        for ch in range(2):
            context=temporal_context(np.where(mask[:,:,ch],x[:,:,ch],np.nan))
            interp[:,:,ch]=np.where(np.isfinite(context[4]),context[4],means[None,:,ch])
            if ch==0:
                span=context[2]+context[3]
                gap=np.stack([np.where(np.isfinite(span),np.clip(span,0,120)/120,1),~np.isfinite(span)],-1).astype(np.float32)
        ramp_net,ramp_valid=ramp_grid(pd.read_parquet(ramps),date,nodes,attachments)
        if not np.isfinite(interp).all():
            raise ValueError('Interpolation fallback is nonfinite')
        np.savez_compressed(destination/'inputs.npz',interp=interp,gap=gap,ramp_net=ramp_net,ramp_valid=ramp_valid)
        return dict(date=date,shape=list(x.shape),wave_folder=str(wave_folder),grid_folder=str(grid_folder),truth_used=False,
                    valid_ramp_cells=int(ramp_valid.sum()))
    folder,manifest,hit=cache.get_or_build(identity,build)
    return dict(date=date,key=manifest['key'],hit=hit,folder=str(folder),grid_folder=str(grid_folder),wave_folder=str(wave_folder),
                wave_key=wave_manifest['key'],grid_key=grid_manifest['key'])


def make_input(x,mask,interp,gap,wave,means,mean,std,date,attributes,use_wave):
    if x.shape!=mask.shape or x.shape!=interp.shape:
        raise ValueError('Daily input shape differs')
    visible=np.where(mask,x,0).astype(np.float32)
    normalized=np.where(mask,(visible-mean)/std,0)
    n=x.shape[1]
    common=[normalized,mask.astype(np.float32),(interp-mean)/std,gap,time_covariates(date,n),
            np.broadcast_to(attributes[None],(len(x),*attributes.shape))]
    if use_wave:
        raw=wave['mixed'];fallback=~np.isfinite(raw)
        prior=np.where(fallback,means[None],raw).astype(np.float32)
        extras=[]
        for name in ['free','congested','mixed']:
            values=np.where(np.isfinite(wave[name]),wave[name],means[None])
            extras.append((values-mean)/std)
        extras += [wave['coverage_free'],wave['coverage_congested'],wave['weight'][...,None],fallback.astype(np.float32)]
        extra=np.concatenate(extras,-1)
    else:
        prior=interp.copy();extra=np.zeros((*x.shape[:-1],13),np.float32)
    features=np.concatenate(common+[extra],-1).astype(np.float32)
    if features.shape[-1]!=31 or not np.isfinite(features).all() or not np.isfinite(prior).all():
        raise ValueError('Nonfinite features or incorrect schema')
    return features,np.maximum(prior,0),visible


class Windows(Dataset):
    def __init__(self,records,grid_metadata,means,attributes,use_wave,training,window=48,stride=24):
        self.records={r['date']:r for r in records};self.dates=sorted(self.records)
        self.mean=np.asarray(grid_metadata['scaler']['mean'],np.float32)
        self.std=np.asarray(grid_metadata['scaler']['std'],np.float32)
        self.means,self.attributes,self.use_wave,self.training=means,attributes,use_wave,training
        self.window=window
        self.items=[(date,start) for date in self.dates for start in window_starts(window,stride)]

    @lru_cache(maxsize=8)
    def day(self,date):
        record=self.records[date]
        with np.load(Path(record['grid_folder'])/(date+'.npz'),allow_pickle=False) as a:
            x,mask=a['x'],a['mask']
            if self.training:
                y,supervise=a['y'],a['supervise']
            else:
                y,supervise=np.zeros_like(x),np.zeros_like(mask)
        if np.any(supervise & mask):
            raise ValueError('A supervised hidden target is visible')
        with np.load(Path(record['folder'])/'inputs.npz',allow_pickle=False) as a:
            interp,gap,ramps,valid=(a[k] for k in ['interp','gap','ramp_net','ramp_valid'])
        wave=None
        if self.use_wave:
            with np.load(Path(record['wave_folder'])/'wave.npz',allow_pickle=False) as a:
                wave={k:a[k] for k in a.files}
        features,prior,visible=make_input(x,mask,interp,gap,wave,self.means,self.mean,self.std,date,self.attributes,self.use_wave)
        return dict(features=features,prior=prior,x=visible,mask=mask,y=y,supervise=supervise,ramps=ramps,ramp_valid=valid)

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        date,start=self.items[index]
        return {k:torch.from_numpy(v[start:start+self.window].copy()) for k,v in self.day(date).items()}
