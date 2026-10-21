"""Complementary local flow expert. Training/development only until frozen.

Adds explicit observed temporal offsets to the existing 64 channels. Labels
and masks are used only to select supervised fit/development examples.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import uniform_filter1d
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from optimize import ROOT,CACHE,V1,save,sha

PRIORITY=['D12_I5_N','D7_I405_N','D7_I210_E']
CHANNELS=[0,1,4,5,6,7,48,49]
SHIFTS=[-12,-6,-3,-1,1,3,6,12]
FOLDER=ROOT/'local-flow'
FEATURES=134
PROFILE_CACHE={}


def feature_grid(x,history):
    """64 base, 64 offsets, 3 fit-only history, 3 observed profile adjustments."""
    source=x[CHANNELS].astype(np.float32)
    pieces=[x.transpose(1,2,0).astype(np.float32)]
    for shift in SHIFTS:
        z=np.zeros_like(source)
        if shift>0:z[:,shift:]=source[:,:-shift]
        else:z[:,:shift]=source[:,-shift:]
        pieces.append(z.transpose(1,2,0))
    pieces.append(history.transpose(1,2,0))
    mean=history[0].astype(np.float32);visible=x[6]>0
    flow=x[5].astype(np.float32)*(2000/600)
    observed=np.where(visible,flow,0).sum(0)
    expected=np.where(visible,mean,0).sum(0)
    global_ratio=float(observed.sum()/expected.sum()) if expected.sum()>1e-8 else 1.
    ratio=np.clip((observed+24*global_ratio)/(expected+24),.5,1.5)
    day_profile=mean*ratio[None]
    residual=np.where(visible,flow-day_profile,0)
    local_sum=uniform_filter1d(residual,25,axis=0,mode='constant')*25
    local_count=uniform_filter1d(visible.astype(np.float32),25,axis=0,mode='constant')*25
    local_profile=np.maximum(0,day_profile+local_sum/(local_count+4))
    pieces.append(np.stack([day_profile,local_profile,local_count/25],axis=-1))
    return np.concatenate(pieces,-1).astype(np.float16)


def profile_features(sums,squares,counts):
    mean=sums/np.maximum(counts,1)
    std=np.sqrt(np.maximum(0,squares/np.maximum(counts,1)-mean**2))
    return np.stack([mean,std,np.minimum(counts,32)/32]).astype(np.float16)


def history_for_day(panel,day):
    if panel not in PROFILE_CACHE:
        PROFILE_CACHE[panel]=np.load(FOLDER/panel/'fit-profiles.npz')['profiles']
    return PROFILE_CACHE[panel][__import__('datetime').date.fromisoformat(day).weekday()]


class Block(nn.Module):
    def __init__(self,w):
        super().__init__();self.norm=nn.LayerNorm(w)
        self.net=nn.Sequential(nn.Linear(w,w*2),nn.SiLU(),nn.Dropout(.05),nn.Linear(w*2,w))
        self.scale=nn.Parameter(torch.full((w,),.1))
    def forward(self,x):return x+self.net(self.norm(x))*self.scale


class FlowExpert(nn.Module):
    def __init__(self,nodes,width=512):
        super().__init__();self.node=nn.Embedding(nodes,32)
        self.encode=nn.Linear(FEATURES+32,width);self.blocks=nn.ModuleList([Block(width) for _ in range(4)])
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,node):
        z=self.encode(torch.cat([x,self.node(node)],-1))
        for block in self.blocks:
            z=checkpoint(block,z,use_reentrant=False) if self.training else block(z)
        return (x[:,1].float()*(2000/600)+self.head(z)[:,0].float()).clamp_min(0)


def predict_grid(model,x,panel,day):
    z=feature_grid(x,history_for_day(panel,day));t,n,_=z.shape;z=z.reshape(-1,FEATURES)
    node=np.broadcast_to(np.arange(n,dtype=np.int64),(t,n)).ravel()
    values=[]
    for start in range(0,len(z),32768):
        a=torch.from_numpy(z[start:start+32768].astype(np.float32)).cuda()
        b=torch.from_numpy(node[start:start+32768]).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=model(a,b).float().cpu().numpy()
        assert np.isfinite(p).all();values.append(p)
    return np.concatenate(values).reshape(t,n)


def prepare():
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER/'policy.json').exists():
        revision=json.loads((FOLDER/'policy.json').read_text()).get('revision',1)
        if revision<3:(FOLDER/'policy.json').replace(FOLDER/f'policy-v{revision}.json')
    if not (FOLDER/'policy.json').exists():
        save(FOLDER/'policy.json',dict(recorded_at=__import__('datetime').datetime.now().astimezone().isoformat(),
            revision=3,
            holdout_used=False,panels=PRIORITY,reason='Complement long-context model with an explicit multi-scale local flow expert on the three regressed dense corridors.',
            features='Original64 +8channels at offsets +/-5,15,30,60minutes +fit-period weekday flow mean/std/support +day-adjusted profile/local-adjusted profile/local observation fraction; 134total; 32 learned node dimensions.',
            adjustment='Day amplitude ratio only from same-day visible values, regularization24, clipped[.5,1.5]; local residual correction uniform25ticks with weight4. No hidden target values read.',
            profile='Only fitting indices28:196. Each fit day is subtracted from its own profile. Dev/holdout/competition features use fitting-period statistics only. No dev or holdout labels in feature profiles.',
            architecture='width512 four residual MLP blocks expansion2; interpolation residual; flow output only',
            fit_indices=[28,196],development_indices=[196,218],training_steps=300,checkpoints=[100,200,300],
            loss='flow MSE in units of 600 veh/hour/lane',seed=20261025,
            mixing=[0,.05,.1,.2,.3,.5,.75,1.],selection='Internal development only; require pooled and regime-macro RMSE improve in both early and late halves. Then choose lowest full development pooled RMSE.',
            source_sha256=sha(Path(__file__))))
    metadata=json.loads((CACHE/'metadata.json').read_text())
    for pm in metadata['panels']:
        panel=pm['panel']
        if panel not in PRIORITY:continue
        out=FOLDER/panel;out.mkdir(exist_ok=True)
        if (out/'prepared.json').exists():
            if json.loads((out/'prepared.json').read_text())['features']==FEATURES:continue
            previous_features=json.loads((out/'prepared.json').read_text())['features']
            (out/'prepared.json').replace(out/f'prepared-{previous_features}.json')
        old=Path(pm['old_cache']);x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
        y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
        sums=np.zeros((7,288,pm['nodes']),np.float64);squares=np.zeros_like(sums);counts=np.zeros_like(sums)
        weekdays=[__import__('datetime').date.fromisoformat(d).weekday() for d in pm['dates']]
        for d in range(28,196):
            value=np.array(y[d,1],np.float64);valid=np.isfinite(value)&(value>0)
            value=np.where(valid,value,0);w=weekdays[d]
            sums[w]+=value;squares[w]+=value**2;counts[w]+=valid
        profiles=np.stack([profile_features(sums[w],squares[w],counts[w]) for w in range(7)])
        np.savez_compressed(out/'fit-profiles.npz',profiles=profiles)
        for name,days in [('fit',range(28,196)),('dev',range(196,218))]:
            features=[];labels=[];nodes=[];day_ids=[]
            for d in days:
                w=weekdays[d]
                if name=='fit':
                    value=np.array(y[d,1],np.float64);valid=np.isfinite(value)&(value>0);value=np.where(valid,value,0)
                    history=profile_features(sums[w]-value,squares[w]-value**2,counts[w]-valid)
                else:history=profiles[w]
                m=mask[d];z=feature_grid(x[d],history);features.append(z[m]);labels.append(y[d,1][m])
                nodes.append(np.broadcast_to(np.arange(pm['nodes'],dtype=np.int16),m.shape)[m])
                day_ids.append(np.full(int(m.sum()),d,np.int16))
            for label,parts in [('x',features),('y',labels),('nodes',nodes),('days',day_ids)]:np.save(out/f'{name}-{label}.npy',np.concatenate(parts))
        save(out/'prepared.json',dict(panel=panel,nodes=pm['nodes'],features=FEATURES,profile='fit days28:196 only; leave-own-day-out for fitting rows'))
        print('Prepared '+panel,flush=True)


def worker(panel):
    torch.set_num_threads(4);torch.manual_seed(20261025)
    torch.cuda.set_per_process_memory_fraction(.89)
    began=time.monotonic();out=FOLDER/panel
    pm=next(p for p in json.loads((CACHE/'metadata.json').read_text())['panels'] if p['panel']==panel)
    nodes=pm['nodes'];model=FlowExpert(nodes).cuda();ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    fitx=torch.from_numpy(np.load(out/'fit-x.npy')).cuda()
    fity=torch.from_numpy(np.load(out/'fit-y.npy').astype(np.float32)).cuda()
    fitnode=torch.from_numpy(np.load(out/'fit-nodes.npy').astype(np.int64)).cuda()
    dx=np.load(out/'dev-x.npy');dy=np.load(out/'dev-y.npy').astype(np.float64)
    dn=np.load(out/'dev-nodes.npy');dd=np.load(out/'dev-days.npy')
    regimes=np.array([pm['regimes'][d] for d in dd])
    # First-stage channel blend was already selected without holdout.
    selected=np.load(ROOT/panel/'selected-development.npz')['prediction']
    old=Path(pm['old_cache']);masks=np.load(old/'target.npy',mmap_mode='r')
    previous=np.concatenate([selected[i,1][masks[d]] for i,d in enumerate(range(196,218))]).astype(np.float64)
    def metrics(pred,subset):
        err=(pred-dy)**2
        pool=float(np.sqrt(err[subset].mean())*600)
        macro=float(np.mean([np.sqrt(err[subset&(regimes==r)].mean())*600 for r in ['R1','R2','R3']]))
        return np.array([pool,macro])
    groups=[np.ones(len(dy),bool),dd<207,dd>=207]
    baseline=[metrics(previous,g) for g in groups]
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    def predict():
        values=[]
        for start in range(0,len(dx),32768):
            z=torch.from_numpy(dx[start:start+32768].astype(np.float32)).cuda()
            n=torch.from_numpy(dn[start:start+32768].astype(np.int64)).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=ema(z,n).cpu().numpy()
            assert np.isfinite(p).all();values.append(p)
        return np.concatenate(values)
    # Probe includes backward and optimizer states before choosing actual batch.
    batch=32768
    optimizer.zero_grad(set_to_none=True)
    for candidate in [32768,65536,131072]:
        try:
            ids=torch.randint(len(fity),(candidate,),device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):p=model(fitx[ids].float(),fitnode[ids]);loss=(p-fity[ids]).square().mean()
            loss.backward();torch.cuda.synchronize()
            allocated=torch.cuda.max_memory_allocated()/1024**3
            if allocated>10.5:break
            batch=candidate
        except torch.OutOfMemoryError:break
        finally:
            ids=p=loss=None;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    best=baseline[0][0];choice=dict(alpha=0.,step=0,pooled_flow=best,macro_flow=baseline[0][1],parameters=sum(p.numel() for p in model.parameters()),batch=batch)
    history=[]
    for step in range(1,301):
        model.train();optimizer.zero_grad(set_to_none=True)
        ids=torch.randint(len(fity),(batch,),device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):p=model(fitx[ids].float(),fitnode[ids]);loss=(p-fity[ids]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite flow-expert loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        rate=2e-4*(.1+.9*(1+np.cos(np.pi*step/300))/2)
        for group in optimizer.param_groups:group['lr']=rate
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.05)
        if step%25==0:
            progress=dict(panel=panel,step=step,batch=batch,loss=float(loss.detach()),seconds=time.monotonic()-began)
            save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step%100==0:
            raw=predict();candidates=[]
            for alpha in [0,.05,.1,.2,.3,.5,.75,1.]:
                value=(1-alpha)*previous+alpha*raw
                values=[metrics(value,g) for g in groups]
                if all(np.all(a<=b+1e-9) for a,b in zip(values,baseline)):
                    candidates.append((values[0][0],alpha,values[0][1]))
            score,alpha,macro=min(candidates)
            if score<best-.02:
                best=score;choice=dict(alpha=alpha,step=step,pooled_flow=score,macro_flow=macro,parameters=choice['parameters'],batch=batch)
                torch.save(dict(state_dict=ema.state_dict(),nodes=nodes),out/'best-model.pt')
                np.save(out/'selected-development-flow.npy',((1-alpha)*previous+alpha*raw).astype(np.float32))
            history.append(dict(step=step,best=choice.copy(),raw_pooled_flow=metrics(raw,groups[0])[0]))
            save(out/'history.json',history)
    choice.update(panel=panel,seconds=time.monotonic()-began,holdout_used=False)
    save(out/'selection.json',choice);print(json.dumps(choice),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');args=p.parse_args()
    prepare() if args.prepare else worker(args.worker)
