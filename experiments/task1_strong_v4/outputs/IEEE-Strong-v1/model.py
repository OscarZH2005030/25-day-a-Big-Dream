"""Gap-aware bidirectional diagonal-SSM + multi-hop graph mixer.

An independent architecture inspired by diagonal SSMs, not a reproduction of
S4D/Mamba and not the team's graph Transformer. No attention module is used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def propagate(x, index, weight):
    b,t,n,c=x.shape
    gathered=x.index_select(2,index.flatten()).reshape(b,t,n,index.shape[1],c)
    return (gathered*weight[None,None,:,:,None]).sum(3)


class BidirectionalDiagonalSSM(nn.Module):
    def __init__(self,width,rank=128,states=16):
        super().__init__()
        self.encode=nn.Linear(width,rank)
        self.decode=nn.Linear(rank,width)
        self.gate=nn.Linear(width,width)
        rates=1/torch.logspace(math.log10(2),math.log10(288),states)
        self.log_rate=nn.Parameter(rates.log().repeat(rank,1))
        self.forward_coeff=nn.Parameter(torch.randn(rank,states)/math.sqrt(states))
        self.backward_coeff=nn.Parameter(torch.randn(rank,states)/math.sqrt(states))
        self.skip=nn.Parameter(torch.ones(rank))

    def forward(self,x):
        z=self.encode(x)
        length=z.shape[1]
        fft_length=1<<(2*length-1).bit_length()
        with torch.autocast('cuda',enabled=False):
            t=torch.arange(length,device=x.device,dtype=torch.float32)
            basis=torch.exp(-self.log_rate.float().exp()[...,None]*t)
            kf=(self.forward_coeff.float()[...,None]*basis).sum(1)
            kb=(self.backward_coeff.float()[...,None]*basis).sum(1)
            kf=kf/kf.abs().sum(-1,keepdim=True).clamp_min(1e-3)
            kb=kb/kb.abs().sum(-1,keepdim=True).clamp_min(1e-3)
            signal=z.float().permute(0,2,3,1)
            frequency=torch.fft.rfft(signal,n=fft_length)
            forward=torch.fft.irfft(frequency*torch.fft.rfft(kf,n=fft_length),n=fft_length)[...,:length]
            reverse=torch.fft.rfft(signal.flip(-1),n=fft_length)
            backward=torch.fft.irfft(reverse*torch.fft.rfft(kb,n=fft_length),n=fft_length)[...,:length].flip(-1)
            mixed=(forward+backward+signal*self.skip.float()[None,None,:,None]).permute(0,3,1,2)
        return self.decode(F.silu(mixed.to(z.dtype)))*torch.sigmoid(self.gate(x))


class GraphStateBlock(nn.Module):
    def __init__(self,width,dilation,global_state=False,dropout=.05):
        super().__init__()
        self.time_norm=nn.LayerNorm(width)
        self.temporal=nn.Conv2d(width,width,(7,1),padding=(3*dilation,0),
                                dilation=(dilation,1),groups=width)
        self.time_project=nn.Linear(width,width)
        self.graph_norm=nn.LayerNorm(width)
        self.graph_project=nn.Linear(5*width,width)
        self.edge_gate=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,4),nn.Sigmoid())
        self.state_norm=nn.LayerNorm(width) if global_state else None
        self.state=BidirectionalDiagonalSSM(width) if global_state else None
        self.ff_norm=nn.LayerNorm(width)
        self.ff_in=nn.Linear(width,6*width)
        self.ff_out=nn.Linear(3*width,width)
        self.drop=nn.Dropout(dropout)
        self.scale_time=nn.Parameter(torch.full((width,),.1))
        self.scale_graph=nn.Parameter(torch.full((width,),.1))
        self.scale_ff=nn.Parameter(torch.full((width,),.1))

    def forward(self,x,context,up_index,up_weight,down_index,down_weight):
        z=self.time_norm(x)
        z=self.temporal(z.permute(0,3,1,2)).permute(0,2,3,1)
        x=x+self.drop(self.time_project(F.silu(z)))*self.scale_time
        z=self.graph_norm(x)
        up=propagate(z,up_index,up_weight);down=propagate(z,down_index,down_weight)
        up2=propagate(up,up_index,up_weight);down2=propagate(down,down_index,down_weight)
        gates=self.edge_gate(context)
        joined=torch.cat([z,up*gates[...,0,None],down*gates[...,1,None],
                          up2*gates[...,2,None],down2*gates[...,3,None]],-1)
        x=x+self.drop(self.graph_project(joined))*self.scale_graph
        if self.state is not None:
            x=x+self.drop(self.state(self.state_norm(x)))*.1
        a,b=self.ff_in(self.ff_norm(x)).chunk(2,-1)
        x=x+self.drop(self.ff_out(F.silu(a)*b))*self.scale_ff
        return x


class GapGraphSSM(nn.Module):
    def __init__(self,features=48,width=256,depth=10,dropout=.05,checkpoint_blocks=True):
        super().__init__()
        self.width,self.depth=width,depth
        self.checkpoint_blocks=checkpoint_blocks
        self.encode=nn.Linear(features,width)
        self.node_embedding=nn.Embedding(2048,width)
        nn.init.normal_(self.node_embedding.weight,std=.02)
        dilations=[1,2,4,8,16,32,1,2,4,8]
        self.blocks=nn.ModuleList([GraphStateBlock(width,dilations[i%10],i%3==2,dropout) for i in range(depth)])
        self.local=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))
        self.global_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2))
        self.expert_gate=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,2))
        self.register_buffer('base_scale',torch.tensor([4.,2000/600]))
        for head in [self.local[-1],self.global_head[-1]]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def forward(self,features,graph):
        # features: [B,C,T,N]; all graph nodes of one corridor are kept together.
        inputs=features.permute(0,2,3,1)
        base=inputs[...,:2].float()*self.base_scale
        context=inputs[...,[6,7,12,13,14,15,16,19,22,47]]
        x=self.encode(inputs)
        if 'node_id' in graph:
            x=x+self.node_embedding(graph['node_id'])[None,None]
        local=None
        args=[graph[k] for k in ['up_index','up_weight','down_index','down_weight']]
        for i,block in enumerate(self.blocks):
            if self.training and self.checkpoint_blocks:
                x=checkpoint(block,x,context,*args,use_reentrant=False)
            else:
                x=block(x,context,*args)
            if i==1:local=self.local(x).float()
        if local is None:local=self.local(x).float()
        gate=torch.sigmoid(self.expert_gate(context)).float()
        residual=(1-gate)*local+gate*self.global_head(x).float()
        return (base+residual).clamp_min(0).permute(0,3,1,2)
