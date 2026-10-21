"""Function-preserving width expansion of our trained graph/SSM/attention model."""
import sys
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'IEEE-Strong-v1'))
from hybrid_model import HybridGraphSSM,FullDayAttention


class WideGraphSSM(HybridGraphSSM):
    def __init__(self,features=64,width=512,depth=10,dropout=.05,checkpoint_blocks=True):
        assert width==512 and depth==10
        super().__init__(features,width,depth,dropout,checkpoint_blocks)
        # Duplicate the eight original heads, retaining a 32-channel head.
        self.time_attention=nn.ModuleDict({str(i):FullDayAttention(width,16,dropout,
            checkpoint_chunks=checkpoint_blocks) for i in [4,9]})


def expand_state(state,symmetry_noise=.01,seed=20261027):
    """Duplicate activations and halve incoming coefficients for duplicated inputs.

    Opposite perturbations on the two incoming copies cancel when activations
    are identical, while allowing their gradients to diverge during learning.
    """
    generator=torch.Generator(device='cpu').manual_seed(seed)
    def linear(t,out_group=None,in_group=None,noise=True):
        t=t.cpu().clone()
        if out_group is not None:
            t=torch.cat([
                torch.cat([z,z],0) for z in t.split(out_group,dim=0)],0)
        if in_group is not None:
            parts=[]
            for z in t.split(in_group,dim=1):
                z=z*.5
                epsilon=torch.randn(z.shape,generator=generator)*float(z.std())*symmetry_noise if noise else torch.zeros_like(z)
                parts.append(torch.cat([z+epsilon,z-epsilon],1))
            t=torch.cat(parts,1)
        return t
    result={}
    for key,tensor in state.items():
        t=tensor.detach().cpu()
        if key=='base_scale':pass
        elif key=='encode.weight':t=linear(t,256)
        elif key=='encode.bias':t=linear(t,256)
        elif key=='node_embedding.weight':t=torch.cat([t,t],1)
        elif key.startswith('expert_gate.') or '.edge_gate.' in key:pass
        elif key.startswith(('local.','global_head.')):
            if '.0.' in key:t=linear(t,256)
            elif key.endswith('weight'):t=linear(t,in_group=256)
        elif key.startswith('time_attention.'):
            if '.qkv.weight' in key:t=linear(t,256,256)
            elif '.qkv.bias' in key:t=linear(t,256)
            elif '.project.weight' in key:t=linear(t,256,256)
            else:t=linear(t,256)
        elif key.startswith('blocks.'):
            if '.state.' in key:
                if '.state.encode.weight' in key:t=linear(t,in_group=256)
                elif '.state.decode.' in key:t=linear(t,256)
                elif '.state.gate.weight' in key:t=linear(t,256,256)
                elif '.state.gate.bias' in key:t=linear(t,256)
            elif key.endswith('graph_project.weight'):t=linear(t,256,256)
            elif key.endswith('ff_in.weight'):t=linear(t,768,256)
            elif key.endswith('ff_in.bias'):t=linear(t,768)
            elif key.endswith('ff_out.weight'):t=linear(t,256,768)
            elif key.endswith('time_project.weight'):t=linear(t,256,256)
            else:t=linear(t,256)
        else:raise KeyError('Unhandled parameter: '+key)
        result[key]=t.contiguous()
    return result


def verify_cpu():
    torch.set_num_threads(4);torch.manual_seed(20261027)
    small=HybridGraphSSM(64,256,10,dropout=0,checkpoint_blocks=False).eval()
    # Nonzero heads exercise every path, including attention and state layers.
    for head in [small.local[-1],small.global_head[-1]]:nn.init.normal_(head.weight,std=.02)
    for a in small.time_attention.values():nn.init.normal_(a.project.weight,std=.01)
    big=WideGraphSSM(dropout=0,checkpoint_blocks=False).eval()
    big.load_state_dict(expand_state(small.state_dict()))
    n=3;x=torch.randn(1,64,17,n)
    graph=dict(up_index=torch.arange(n)[:,None],down_index=torch.arange(n)[:,None],
        up_weight=torch.ones(n,1),down_weight=torch.ones(n,1),node_id=torch.arange(n))
    with torch.inference_mode():difference=(small(x,graph)-big(x,graph)).abs().max().item()
    assert difference<2e-5,difference
    return dict(passed=True,maximum_float32_difference=difference,
        small_parameters=sum(p.numel() for p in small.parameters()),
        wide_parameters=sum(p.numel() for p in big.parameters()))


if __name__=='__main__':print(verify_cpu())
