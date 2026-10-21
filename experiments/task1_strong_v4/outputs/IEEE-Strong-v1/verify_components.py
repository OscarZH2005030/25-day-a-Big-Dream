"""Independent CPU numerical checks for the new state-space and graph operators."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from model import BidirectionalDiagonalSSM,propagate

torch.manual_seed(17)
torch.set_num_threads(2)
layer=BidirectionalDiagonalSSM(16,rank=8,states=4)
x=torch.randn(2,17,3,16,requires_grad=True)
actual=layer(x)
z=layer.encode(x);b,t,n,r=z.shape
ticks=torch.arange(t,dtype=torch.float32)
basis=torch.exp(-layer.log_rate.exp()[...,None]*ticks)
kf=(layer.forward_coeff[...,None]*basis).sum(1)
kb=(layer.backward_coeff[...,None]*basis).sum(1)
kf=kf/kf.abs().sum(-1,keepdim=True).clamp_min(1e-3)
kb=kb/kb.abs().sum(-1,keepdim=True).clamp_min(1e-3)
signal=z.permute(0,2,3,1).reshape(b*n,r,t)
forward=F.conv1d(F.pad(signal,(t-1,0)),kf.flip(-1)[:,None],groups=r)
backward=F.conv1d(F.pad(signal.flip(-1),(t-1,0)),kb.flip(-1)[:,None],groups=r).flip(-1)
reference=(forward+backward+signal*layer.skip[None,:,None]).reshape(b,n,r,t).permute(0,3,1,2)
reference=layer.decode(F.silu(reference))*torch.sigmoid(layer.gate(x))
error=float((actual-reference).abs().max().detach())
assert error<1e-5,error
actual.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
values=torch.randn(2,5,4,7)
index=torch.tensor([[1,2],[0,2],[1,3],[0,0]])
weight=torch.tensor([[.5,.5],[.25,.75],[.3,.7],[0.,0.]])
dense=torch.zeros(4,4)
for i in range(4):
    for j in range(2):dense[i,index[i,j]]+=weight[i,j]
graph_error=float((propagate(values,index,weight)-torch.einsum('nm,btmc->btnc',dense,values)).abs().max())
assert graph_error<1e-6
result=dict(cpu_only=True,fft_vs_direct_convolution_max_error=error,
            graph_vs_dense_operator_max_error=graph_error,all_ssm_gradients_finite=True)
Path(__file__).with_name('component-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
