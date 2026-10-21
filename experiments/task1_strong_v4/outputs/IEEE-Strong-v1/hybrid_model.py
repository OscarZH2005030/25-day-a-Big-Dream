"""Warm-expand the graph SSM with content-dependent full-day time attention."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model import GapGraphSSM


class FullDayAttention(nn.Module):
    def __init__(self,width,heads=8,dropout=.05,query_chunk=32,checkpoint_chunks=True):
        super().__init__()
        assert width%heads==0
        self.heads=heads;self.head_size=width//heads;self.dropout=dropout
        self.query_chunk=query_chunk;self.checkpoint_chunks=checkpoint_chunks
        self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width)
        self.project=nn.Linear(width,width)
        self.scale=nn.Parameter(torch.full((width,),.1))
        # Exact function preservation when expanding an existing trained model.
        nn.init.zeros_(self.project.weight);nn.init.zeros_(self.project.bias)

    def attend(self,q,k,v):
        return F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.)

    def forward(self,x):
        batch,length,nodes,width=x.shape
        z=self.norm(x).permute(0,2,1,3).reshape(batch*nodes,length,width)
        qkv=self.qkv(z).reshape(batch*nodes,length,3,self.heads,self.head_size)
        q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
        chunks=[]
        for start in range(0,length,self.query_chunk):
            query=q[:,:,start:start+self.query_chunk]
            if self.training and self.checkpoint_chunks:
                value=checkpoint(self.attend,query,k,v,use_reentrant=False)
            else:value=self.attend(query,k,v)
            chunks.append(value)
        # Every query chunk still attends to ALL daily keys/values. This is exact
        # full-day attention, not a local attention window or spatial subpatch.
        z=torch.cat(chunks,dim=2)
        z=z.transpose(1,2).reshape(batch,nodes,length,width).permute(0,2,1,3)
        return x+self.project(z)*self.scale


class HybridGraphSSM(GapGraphSSM):
    def __init__(self,features=48,width=256,depth=10,dropout=.05,checkpoint_blocks=True):
        super().__init__(features,width,depth,dropout,checkpoint_blocks)
        self.time_attention=nn.ModuleDict({str(i):FullDayAttention(width,8,dropout,checkpoint_chunks=checkpoint_blocks)
            for i in sorted(set([depth//2-1,depth-1]))})

    def forward(self,features,graph):
        inputs=features.permute(0,2,3,1)
        base=inputs[...,:2].float()*self.base_scale
        context=inputs[...,[6,7,12,13,14,15,16,19,22,47]]
        x=self.encode(inputs)
        if 'node_id' in graph:x=x+self.node_embedding(graph['node_id'])[None,None]
        local=None
        args=[graph[k] for k in ['up_index','up_weight','down_index','down_weight']]
        for i,block in enumerate(self.blocks):
            if self.training and self.checkpoint_blocks:x=checkpoint(block,x,context,*args,use_reentrant=False)
            else:x=block(x,context,*args)
            if str(i) in self.time_attention:
                attention=self.time_attention[str(i)]
                x=attention(x)
            if i==1:local=self.local(x).float()
        if local is None:local=self.local(x).float()
        gate=torch.sigmoid(self.expert_gate(context)).float()
        residual=(1-gate)*local+gate*self.global_head(x).float()
        return (base+residual).clamp_min(0).permute(0,3,1,2)


def make_model(config,features):
    cls=HybridGraphSSM if config.get('variant')=='full_day_attention_ssm' else GapGraphSSM
    return cls(features,config['width'],config['depth'])


def load_warm_state(model,state):
    state=dict(state)
    old=state['encode.weight']
    extra=model.encode.weight.shape[1]-old.shape[1]
    assert extra>=0
    if extra:state['encode.weight']=F.pad(old,(0,extra))
    status=model.load_state_dict(state,strict=False)
    assert not status.unexpected_keys and all(k.startswith('time_attention.') for k in status.missing_keys)
    return status
