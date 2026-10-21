"""Flow error correction on the immutable v2 predictor. No Transformer dependency."""
import datetime as dt
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'IEEE-Strong-v2'
sys.path.insert(0,str(BASE))
from optimize import CACHE,V1,WORKSPACE,save,sha,telemetry,FIELDS
from local_flow_expert import Block,FlowExpert,feature_grid,history_for_day,predict_grid as local_predict
from hybrid_model import make_model

PANELS=['D12_I5_N','D7_I405_N','D7_I210_E']
FEATURES=136


class ResidualFlow(nn.Module):
    def __init__(self,nodes,width=512):
        super().__init__();self.node=nn.Embedding(nodes,32)
        self.encode=nn.Linear(FEATURES+32,width)
        self.blocks=nn.ModuleList([Block(width) for _ in range(4)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,node,base):
        z=self.encode(torch.cat([x,self.node(node)],-1))
        for block in self.blocks:
            z=checkpoint(block,z,use_reentrant=False) if self.training else block(z)
        return (base.float()+self.head(z)[:,0].float()).clamp_min(0)


def contexts(x,raw_speed,base_flow,panel,day):
    local=feature_grid(x,history_for_day(panel,day))
    return np.concatenate([local,raw_speed[...,None],base_flow[...,None]],axis=-1).astype(np.float16)


def infer_rows(model,features,nodes,base):
    values=[]
    for start in range(0,len(base),32768):
        z=torch.from_numpy(np.array(features[start:start+32768],np.float32)).cuda()
        n=torch.from_numpy(np.array(nodes[start:start+32768],np.int64)).cuda()
        b=torch.from_numpy(np.array(base[start:start+32768],np.float32)).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=model(z,n,b).float().cpu().numpy()
        assert np.isfinite(p).all();values.append(p)
    return np.concatenate(values)


def base_model(pm):
    manifest=json.loads((BASE/'final-model.json').read_text())
    route=next(r for r in manifest['routes'] if r['panel']==pm['panel'])
    assert route['checkpoint'] and route['blend'][1]==1., 'Preparation expects v2 raw flow base for these three routes'
    state=torch.load(BASE/pm['panel']/route['checkpoint'],map_location='cuda',weights_only=True)
    model=make_model(state['config'],64).cuda().eval();model.load_state_dict(state['state_dict'])
    graph={k:torch.from_numpy(v).cuda() for k,v in np.load(V1/route['previous_route']['model_graph']).items()}
    metadata=json.loads((CACHE/'metadata.json').read_text());offset=0
    for p in metadata['panels']:
        if p['panel']==pm['panel']:break
        offset+=p['nodes']
    graph['node_id']=torch.arange(offset,offset+pm['nodes'],device='cuda')
    local=next((r for r in manifest['local_flow_routes'] if r['panel']==pm['panel']),None)
    booster=None
    if local and local['alpha']:
        state=torch.load(BASE/'local-flow'/pm['panel']/'best-model.pt',map_location='cuda',weights_only=True)
        booster=FlowExpert(pm['nodes']).cuda().eval();booster.load_state_dict(state['state_dict'])
    return model,graph,booster,float(local['alpha']) if local else 0.


def prepare(panel):
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.89)
    out=ROOT/panel;out.mkdir(exist_ok=True)
    pm=next(p for p in json.loads((CACHE/'metadata.json').read_text())['panels'] if p['panel']==panel)
    model,graph,booster,alpha=base_model(pm)
    x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
    old=Path(pm['old_cache']);masks=np.load(old/'target.npy',mmap_mode='r')
    for split,days in [('fit',range(28,196)),('dev',range(196,218))]:
        source=BASE/'local-flow'/panel
        old_features=np.load(source/f'{split}-x.npy',mmap_mode='r')
        total=len(old_features)
        dest=np.lib.format.open_memmap(out/f'{split}-features.npy',mode='w+',dtype=np.float16,shape=(total,FEATURES))
        base=np.lib.format.open_memmap(out/f'{split}-base.npy',mode='w+',dtype=np.float32,shape=(total,))
        cursor=0
        for d in days:
            z=np.array(x[d],np.float32)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                raw=model(torch.from_numpy(z[None]).cuda(),graph)[0].float().cpu().numpy()
            flow=raw[1].astype(np.float64)
            if booster is not None:
                extra=local_predict(booster,x[d],panel,pm['dates'][d])
                flow=(1-alpha)*flow+alpha*extra
            m=masks[d];n=int(m.sum());sl=slice(cursor,cursor+n)
            dest[sl,:134]=old_features[sl]
            dest[sl,134]=raw[0][m];dest[sl,135]=flow[m]
            base[sl]=flow[m]
            cursor+=n
            if d%20==0:
                save(out/'progress.json',dict(phase='preparing_frozen_predictions',panel=panel,split=split,day=d))
                print(json.dumps(dict(panel=panel,split=split,day=d)),flush=True)
        assert cursor==total
        dest.flush();base.flush()
    save(out/'prepared.json',dict(nodes=pm['nodes'],features=FEATURES,source=str(source),
        fit_indices=[28,196],dev_indices=[196,218],flow_base='Immutable v2 final model, including its selected local branch',
        caution='Frozen main model was fitted on the fitting period; residual fitting uses its in-sample errors. Development remains outside both models training.'))


def flow_metrics(pred,truth,days,regimes,subset):
    squared=(pred.astype(np.float64)-truth)**2
    pooled=float(np.sqrt(squared[subset].mean())*600)
    macro=float(np.mean([np.sqrt(squared[subset&(regimes==r)].mean())*600 for r in ['R1','R2','R3']]))
    return np.array([pooled,macro])
