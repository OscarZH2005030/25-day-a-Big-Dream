from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from task1.cache import snapshot
from task1.transformer_priority.common import SCALE,sha256
from task1.transformer_priority.data import sampling_spec,graph_tensors
from .augment import (COLUMNS,features_from_observations,link_groups,additional_mask,hide_observations)
from .common import AUGMENT


class Observations:
    def __init__(self,data):
        self.data=data;self.folders={};self.memory=OrderedDict();self.records={}

    def ensure(self,date):
        if date not in self.data.roles['training']:
            raise ValueError('Augmentation may read training dates only')
        if date not in self.folders:
            source=snapshot([self.data.exp.masked_files[date]])
            identity=dict(kind='t1_training_observations_v1',panel=self.data.panel,date=date,
                          source=source,features=self.data.records[date]['key'],columns=COLUMNS,
                          implementation=sha256(Path(__file__)))
            def build(out):
                frame=pd.read_parquet(self.data.exp.masked_files[date])
                values=np.stack([self.data.exp.grid(frame,c) for c in COLUMNS],axis=-1)
                if np.isinf(values).any(): raise ValueError('Infinite published observations')
                np.save(out/'observations.npy',values,allow_pickle=False)
                return dict(date=date,shape=list(values.shape),masked_only=True)
            folder,manifest,hit=self.data.cache.get_or_build(identity,build)
            self.folders[date]=folder
            self.records[date]=dict(folder=str(folder),key=manifest['key'],hit=hit,source=source,
                                    manifest_sha256=sha256(folder/'manifest.json'))
        return self.folders[date]

    def get(self,date):
        if date not in self.memory:
            self.memory[date]=np.load(self.ensure(date)/'observations.npy',allow_pickle=False)
            if len(self.memory)>12: self.memory.popitem(last=False)
        self.memory.move_to_end(date)
        return self.memory[date]


def choose_supervision(original,artificial,labels,extra_values,rng,fraction):
    original=np.asarray(original,bool);artificial=np.asarray(artificial,bool)&~original
    count=int(original.sum())
    extra_count=min(int(np.floor(count*fraction)),int(artificial.sum()))
    target=original.copy();label=labels.copy()
    if extra_count:
        removed=rng.choice(np.flatnonzero(original),extra_count,replace=False)
        added=rng.choice(np.flatnonzero(artificial),extra_count,replace=False)
        target.flat[removed]=False;target.flat[added]=True
        label.reshape(-1,2)[added]=extra_values.reshape(-1,2)[added]
    if target.sum()!=count or (target&original).sum()<count*.8-1e-8:
        raise ValueError('Dynamic supervision budget changed')
    return target,label


class Sampler:
    def __init__(self,data,variant,mapping):
        self.data,self.variant,self.mapping=data,variant,mapping
        self.by_regime={r:[d for d in data.roles['training'] if mapping[d]==r] for r in ['R1','R2','R3']}
        if any(not dates for dates in self.by_regime.values()): raise ValueError('Empty training regime')
        self.observations=Observations(data)
        self.node_unit,self.adjacency=link_groups(data.exp)

    def dynamic_block(self,date,start,nodes,window,rng):
        data=self.data;obs=self.observations.get(date)
        hidden,eligible,info=additional_mask(obs,self.node_unit,self.adjacency,rng,AUGMENT)
        augmented=hide_observations(obs,hidden)
        raw,base,visible,reliability,names=features_from_observations(data.exp,date,augmented,nodes)
        if names!=data.names: raise ValueError('Augmented feature order differs')
        ticks=np.arange(start,start+window);valid=(ticks>=0)&(ticks<288);clipped=ticks.clip(0,287)
        raw=raw[clipped].copy();raw[~valid]=np.nan
        finite=np.isfinite(raw)
        features=np.concatenate([np.where(finite,(raw-data.mean)/data.scale,0),finite],axis=-1).astype(np.float32)
        original=data.array(date,'target')[clipped][:,nodes].copy();original[~valid]=False
        labels=data.label_grid(date)[clipped][:,nodes].copy();labels[~valid]=0
        extra=(hidden&eligible).T[clipped][:,nodes].copy();extra[~valid]=False
        values=np.swapaxes(obs[nodes,:,:2],0,1)[clipped].copy()
        values[:,:,1]/=data.exp.lanes[nodes,None].T
        values=(values/SCALE).astype(np.float32)
        target,label=choose_supervision(original,extra,labels,values,rng,AUGMENT['extra_supervision_fraction'])
        result=dict(features=features,time_valid=valid,target=target,label=label,original_target=original)
        for name,value in [('base',base),('visible',visible),('reliability',reliability)]:
            value=value[clipped].copy();value[~valid]=0;result[name]=value
        info.update(original_targets=int(original.sum()),retained_targets=int((target&original).sum()),
                    extra_targets=int((target&~original).sum()),supervised_targets=int(target.sum()))
        if np.isfinite(augmented[hidden]).any(): raise ValueError('Artificially hidden values leaked')
        return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in result.items()},info

    def batch(self,rng,aux_rng,config):
        data=self.data;dates=data.roles['training']
        chosen_regime=None
        if self.variant=='B_regime':
            chosen_regime=['R1','R2','R3'][int(rng.integers(3))];dates=self.by_regime[chosen_regime]
        nodes,samples=sampling_spec(rng,data.n,data.graph['order'],dates,config['window'],config['patch_nodes'],config['batch_size'],True)
        parts=[];infos=[]
        for date,start in samples:
            if self.variant=='A_dynamic':
                part,info=self.dynamic_block(date,start,nodes,config['window'],aux_rng)
            else:
                part=data.block(date,start,nodes,config['window'],True)
                part['original_target']=part['target'].clone()
                n=int(part['target'].sum())
                info=dict(pattern='none',original_targets=n,retained_targets=n,extra_targets=0,supervised_targets=n)
            info.update(date=date,regime=self.mapping[date]);parts.append(part);infos.append(info)
        return ({k:torch.stack([p[k] for p in parts]) for k in parts[0]},graph_tensors(data.graph,nodes),
                dict(nodes=nodes.tolist(),samples=samples),infos)
