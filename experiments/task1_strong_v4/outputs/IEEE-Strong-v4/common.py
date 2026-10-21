"""The v4 experiment reads, and never modifies, the current v3 best."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
V3=ROOT.parent/'IEEE-Strong-v3'
sys.path.insert(0,str(V3))
import residual_core as rc
from wide_model import WideGraphSSM,expand_state
BASE=rc.BASE;V1=rc.V1;CACHE=rc.CACHE;WORKSPACE=rc.WORKSPACE
save=rc.save;sha=rc.sha;telemetry=rc.telemetry;FIELDS=rc.FIELDS
PANELS=['D7_I405_N','D12_I5_N','D7_I210_E']


def metadata(panel):
    return next(p for p in json.loads((CACHE/'metadata.json').read_text())['panels'] if p['panel']==panel)


def narrow_model(pm):
    model,graph,extra,alpha=rc.base_model(pm)
    del extra
    return model,graph


def current_dev(pm,mask):
    """Reconstruct exact final v3 targets from its frozen flattened flow array."""
    panel=pm['panel']
    p=np.load(BASE/panel/'selected-development.npz')['prediction'].astype(np.float32)
    flow=np.load(V3/panel/'selected-dev-flow.npy');cursor=0
    for i,d in enumerate(range(196,218)):
        n=int(mask[d].sum());p[i,1][mask[d]]=flow[cursor:cursor+n];cursor+=n
    assert cursor==len(flow)
    return p


def predict(model,x,graph):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        p=model(torch.from_numpy(np.array(x[None],np.float32)).cuda(),graph)[0].float().cpu().numpy()
    assert np.isfinite(p).all()
    return p


def flow_metrics(p,truth,mask,regimes,part=None):
    if part is None:part=np.ones(len(p),bool)
    sq=(p.astype(np.float64)-truth)**2
    pooled=np.sqrt((sq[part]*mask[part]).sum()/mask[part].sum())*600
    group=[np.sqrt((sq[part&(regimes==r)]*mask[part&(regimes==r)]).sum()/mask[part&(regimes==r)].sum())*600 for r in ['R1','R2','R3']]
    return np.array([pooled,np.mean(group)])
