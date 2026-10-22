from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from torch import nn
from .common import LAGS, METHODS, upstream


class Block(nn.Module):
    def __init__(self, width, heads, dropout, reliable=False, delayed=False):
        super().__init__()
        self.heads,self.d=heads,width//heads
        self.tnorm=nn.LayerNorm(width)
        self.temporal=nn.MultiheadAttention(width,heads,dropout=dropout,batch_first=True)
        self.snorm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width)
        self.direction_bias=nn.Embedding(3,heads)
        self.project=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,4*width),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*width,width))
        self.drop=nn.Dropout(dropout)
        self.reliable,self.delayed=reliable,delayed
        if reliable:
            self.reliability=nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,2))
            nn.init.zeros_(self.reliability[-1].weight);nn.init.zeros_(self.reliability[-1].bias)
        if delayed:
            bias=torch.full((len(LAGS),heads),-6.)
            bias[LAGS.index(0)]=0.
            self.lag_bias=nn.Parameter(bias)
        self.diagnostic=None

    def forward(self,x,g,reliability,time_valid,diagnostics=False):
        b,t,n,w=x.shape
        z=self.tnorm(x).permute(0,2,1,3).reshape(b*n,t,w)
        padding=~time_valid[:,None,:].expand(b,n,t).reshape(b*n,t)
        # Omitting an all-false mask also preserves the original numerical path.
        z=self.temporal(z,z,z,key_padding_mask=padding if padding.any() else None,need_weights=False)[0]
        x=x+self.drop(z.reshape(b,n,t,w).permute(0,2,1,3))
        q,k,v=self.qkv(self.snorm(x)).chunk(3,dim=-1)
        q=q.reshape(b,t,n,self.heads,self.d)
        k=k.reshape(b,t,n,self.heads,self.d);v=v.reshape(b,t,n,self.heads,self.d)
        neighbors=g['neighbors'];valid=g['neighbor_valid']
        lags=LAGS if self.delayed else [0]
        keys=[];values=[];masks=[];gates=[]
        for lag in lags:
            ticks=torch.arange(t,device=x.device)+lag
            in_bounds=(ticks>=0)&(ticks<t)
            ticks=ticks.clamp(0,t-1)
            keys.append(k[:,ticks][:,:,neighbors]);values.append(v[:,ticks][:,:,neighbors])
            masks.append(valid[None,None,:,:]&in_bounds[None,:,None,None]&time_valid[:,ticks,None,None])
            if self.reliable:
                query=reliability[:,:,:,None,:].expand(b,t,n,neighbors.shape[1],8)
                source=reliability[:,ticks][:,:,neighbors]
                gate=torch.sigmoid(self.reliability(torch.cat([query,source],dim=-1)))
                gates.append(gate.repeat_interleave(self.heads//2,dim=-1))
        # [batch,time,node,neighbor,lag,head,dim]
        keys=torch.stack(keys,dim=4);values=torch.stack(values,dim=4)
        mask=torch.stack(masks,dim=4)
        logits=(q[:,:,:,None,None]*keys).sum(-1)/math.sqrt(self.d)
        logits=logits+self.direction_bias(g['direction'])[None,None,:,:,None,:]
        if self.reliable:
            gate=torch.stack(gates,dim=4)
            logits=logits+torch.log(gate.clamp_min(1e-6))
        if self.delayed:
            logits=logits+self.lag_bias[None,None,None,None,:,:]
        logits=logits.masked_fill(~mask[...,None],-1e4)
        attention=torch.softmax(logits.flatten(3,4),dim=3).reshape_as(logits)
        # Padded queries are never keys, supervised outputs, or evaluation targets.
        message=(attention[...,None]*values).sum((3,4)).reshape(b,t,n,w)
        message=message*time_valid[:,:,None,None]
        x=x+self.drop(self.project(message))
        if diagnostics:
            real=time_valid[:,:,None,None,None,None]&mask[...,None]
            self.diagnostic=dict(valid_attention_mass=float((attention*real).sum().detach()),
                                 invalid_attention_mass=float((attention*(~mask[...,None])*time_valid[:,:,None,None,None,None]).sum().detach()))
            if self.reliable:
                self.diagnostic['mean_gate']=float(gate[mask].mean().detach())
            if self.delayed:
                lag_mass=(attention*time_valid[:,:,None,None,None,None]).sum((0,1,2,3,5))
                self.diagnostic['lag_mass']={str(lag):float(value) for lag,value in zip(lags,lag_mass.detach().cpu())}
        return x+self.drop(self.ff(x))


class PriorityTransformer(nn.Module):
    def __init__(self,config,method):
        super().__init__()
        if method not in METHODS or config['heads']%2:
            raise ValueError('Invalid method or odd head count')
        w=config['width'];self.input=nn.Linear(102,w)
        self.position=nn.Parameter(torch.randn(1,config['window'],1,w)*.01)
        reliable=method in METHODS[2:];delayed=method==METHODS[3]
        self.blocks=nn.ModuleList([Block(w,config['heads'],config['dropout'],reliable,delayed) for _ in range(config['layers'])])
        self.head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('road_scale',torch.tensor([6.,2.,120.,12000.,120.,600.]))

    def forward(self,batch,graph,diagnostics=False):
        x=self.input(batch['features'])+self.position[:,:batch['features'].shape[1]]
        valid=batch.get('time_valid',torch.ones(x.shape[:2],dtype=torch.bool,device=x.device))
        rel=batch.get('reliability',torch.zeros((*x.shape[:3],8),device=x.device))
        for block in self.blocks:
            x=block(x,graph,rel,valid,diagnostics)
        pred=batch['base']+self.head(x)
        return torch.stack([pred[...,0].clamp_min(.04),pred[...,1].clamp_min(0)],dim=-1)


def model_for(config,method):
    torch.manual_seed(config['seed'])
    original=upstream().worker.StateTransformer(102,config['width'],config['heads'],config['layers'],config['window'],config['dropout'])
    initial={k:v.clone() for k,v in original.state_dict().items()}
    torch.manual_seed(config['seed']+17)
    model=PriorityTransformer(config,method)
    result=model.load_state_dict(initial,strict=False)
    if result.unexpected_keys or any('reliability' not in k and 'lag_bias' not in k for k in result.missing_keys):
        raise ValueError('Shared initialization mismatch')
    for name,value in initial.items():
        if not torch.equal(model.state_dict()[name],value):
            raise ValueError('Shared parameter differs: '+name)
    digest=hashlib.sha256()
    for name,value in initial.items():
        digest.update(name.encode());digest.update(value.numpy().tobytes())
    return model,digest.hexdigest()


def objective(pred,batch):
    mask=batch['target'].bool()
    if not mask.any():
        return pred.sum()*0.
    error=pred[mask]-batch['label'][mask]
    rmse=torch.sqrt((error**2).mean(0)+1e-8)
    return .54*rmse[0]+.46*rmse[1]
