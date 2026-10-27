"""因果局部多任务网络：独立六步 vs 64种完整六步模式，另有状态辅助任务。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
class Residual(nn.Module):
 def __init__(self,width):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.GELU(),nn.Dropout(.05),nn.Linear(width*2,width))
 def forward(self,x):return x+self.net(x)
class StructuredQueue(nn.Module):
 def __init__(self,mean,std,width=128):
  super().__init__();self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.01));self.road=nn.Embedding(8,8);self.condition=nn.Embedding(2,4);self.encode=nn.Sequential(nn.Linear(len(mean)+12,width),nn.LayerNorm(width),nn.GELU());self.state_encode=nn.Sequential(nn.Linear(175+12,width),nn.LayerNorm(width),nn.GELU());self.shared=nn.Sequential(Residual(width),Residual(width));self.independent=nn.ModuleList(nn.Linear(width,6) for _ in range(2));self.pattern=nn.ModuleList(nn.Linear(width,64) for _ in range(2));self.future=nn.Linear(width,12);self.reconstruction=nn.Linear(width,2);self.register_buffer('bits',((torch.arange(64)[:,None]>>torch.arange(6)[None,:])&1).float())
 def embeddings(self,n,road,condition):
  r=self.road(torch.as_tensor(road,device=self.mean.device));c=self.condition(torch.as_tensor(condition,device=self.mean.device));return torch.cat([r,c]).expand(n,-1)
 def forward(self,x,road,condition):
  z=((x-self.mean)/self.std).clamp(-8,8);h=self.shared(self.encode(torch.cat([z,self.embeddings(len(z),road,condition)],-1)));ind=self.independent[condition](h);pat=self.pattern[condition](h);return dict(independent_logits=ind,pattern_logits=pat,independent=ind.sigmoid(),pattern=pat.softmax(-1)@self.bits,future_residual=self.future(h))
 def reconstruct(self,raw,static,road,condition,node_mask):
  # raw为[N,13,13]。遮T-5速度、流量、占有率及可泄漏目标的margin/delta。
  z=raw.clone();z[node_mask,11,0]=1;z[node_mask,11,1:3]=0;z[node_mask,11,4:8]=1;z[node_mask,11,7]=0;z[node_mask,11,9:11]=0;z[node_mask,12,10]=0
  h=self.shared(self.state_encode(torch.cat([z.reshape(len(z),-1),static,self.embeddings(len(z),road,condition)],-1)));return self.reconstruction(h)
def queue_loss(out,y,m,teacher,bits):
 mask=m.float();positive=(y*mask).sum();negative=((1-y)*mask).sum();pos_weight=(negative/(positive+1)).sqrt().clamp(1,12);w=mask*torch.where(y>.5,pos_weight,1.);den=w.sum().clamp_min(1);loss=y.sum()*0;details={}
 for name in ['independent','pattern']:
  p=out[name].clamp(1e-5,1-1e-5);bce=(F.binary_cross_entropy(p,y,reduction='none')*w).sum()/den;intersection=(p*y*mask).sum();union=((p+y-p*y)*mask).sum();iou=(1-(intersection+1e-5)/(union+1e-5))*(positive>0);distill=(F.binary_cross_entropy(p,teacher,reduction='none')*mask).sum()/mask.sum().clamp_min(1);loss=loss+bce+.5*iou+.25*distill;details[name+'_bce']=bce.detach()
 # 部分标签似然：未知未来位允许0或1，不能填成0。
 allowed=~(((bits[None]!=y[:,None])&m[:,None]).any(-1));logits=out['pattern_logits'];nll=torch.logsumexp(logits,-1)-torch.logsumexp(logits.masked_fill(~allowed,-torch.inf),-1);linkmask=m.any(-1);event=(y*mask).any(-1);eventw=torch.where(event,torch.as_tensor(8.,device=y.device),torch.as_tensor(1.,device=y.device));loss=loss+.15*(nll*eventw*linkmask).sum()/(eventw*linkmask).sum().clamp_min(1);return loss,details
