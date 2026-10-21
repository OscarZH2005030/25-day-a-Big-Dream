"""CPU verification: preserve the warm-start function and attention gradients."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from model import GapGraphSSM
from hybrid_model import HybridGraphSSM,load_warm_state

torch.set_num_threads(2);torch.manual_seed(17)
root=Path(__file__).resolve().parent
base=GapGraphSSM().eval()
base.load_state_dict(torch.load(root/'training/best-model.pt',map_location='cpu',weights_only=True)['state_dict'])
hybrid=HybridGraphSSM(features=64).eval()
load_warm_state(hybrid,base.state_dict())
x=torch.rand(1,48,65,5)
extended=torch.cat([x,torch.rand(1,16,65,5)],1)
graph=dict(up_index=torch.arange(5).roll(1)[:,None],down_index=torch.arange(5).roll(-1)[:,None],
           up_weight=torch.ones(5,1),down_weight=torch.ones(5,1),node_id=torch.arange(5))
with torch.inference_mode():
    error=float((hybrid(extended,graph)-base(x,graph)).abs().max())
assert error==0.,error
q,k,v=[torch.randn(2,8,65,32) for _ in range(3)]
manual=(q@k.transpose(-2,-1)/32**.5).softmax(-1)@v
attention_error=float((F.scaled_dot_product_attention(q,k,v)-manual).abs().max())
assert attention_error<2e-6
chunked=torch.cat([F.scaled_dot_product_attention(q[:,:,s:s+32],k,v) for s in range(0,65,32)],2)
chunk_error=float((chunked-manual).abs().max())
assert chunk_error<2e-6
hybrid.train();loss=hybrid(extended,graph).square().mean();loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in hybrid.parameters())
new_gradient=sum(float(p.grad.abs().sum()) for n,p in hybrid.named_parameters()
                 if n.startswith('time_attention.') and p.grad is not None)
assert new_gradient>0
result=dict(passed=True,cpu_only=True,warm_start_max_error=error,
    attention_vs_manual_max_error=attention_error,chunked_full_attention_max_error=chunk_error,
    new_attention_gradient_l1=new_gradient,
    parameters=sum(p.numel() for p in hybrid.parameters()))
(root/'hybrid-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
