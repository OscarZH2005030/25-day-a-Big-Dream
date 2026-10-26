# GitHub原始图注意力类，仅复制类定义；不执行原训练入口。
import math
import torch
from torch import nn

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
