"""Flow-focused continuation. Select on development; freeze before holdout.

All previous model files and the previous submission are read-only inputs.
The two channels may independently fall back to the previous calibrated model.
"""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parent
V1=ROOT.parent/'IEEE-Strong-v1'
WORKSPACE=ROOT.parent.parent
CACHE=WORKSPACE/'work/ieee-strong-cache-64'
sys.path.insert(0,str(V1))
sys.path.insert(0,str(ROOT.parent))
from gpu_stress_test import telemetry,FIELDS,save


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def worker(panel):
    import numpy as np
    import torch
    from hybrid_model import make_model
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    torch.cuda.set_per_process_memory_fraction(.89)
    out=ROOT/panel;out.mkdir(exist_ok=True)
    began=time.monotonic()
    def progress(phase,**kwargs):
        d=dict(panel=panel,phase=phase,seconds=time.monotonic()-began,**kwargs)
        save(out/'progress.json',d);print(json.dumps(d),flush=True)
    route=next(r for r in json.loads((V1/'final-model.json').read_text())['routes'] if r['panel']==panel)
    metadata=json.loads((CACHE/'metadata.json').read_text());offset=0
    for pm in metadata['panels']:
        if pm['panel']==panel:break
        offset+=pm['nodes']
    assert pm['panel']==panel
    old=Path(pm['old_cache'])
    x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
    y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
    graph={k:torch.from_numpy(v).cuda() for k,v in np.load(V1/route['model_graph']).items()}
    graph['node_id']=torch.arange(offset,offset+pm['nodes'],device='cuda')
    state=torch.load(V1/route['checkpoint'],map_location='cuda',weights_only=True)
    model=make_model(state['config'],len(metadata['features'])).cuda()
    model.load_state_dict(state['state_dict'])
    config=copy.deepcopy(state['config']);config['phase']='flow-focused Pareto continuation'
    alpha=np.array(json.loads((V1/route['calibration']).read_text())['coefficients'])[:,None,None]
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    dev_days=list(range(196,218))
    target=np.stack([mask[d] for d in dev_days])
    truth=np.stack([y[d] for d in dev_days]).astype(np.float64)
    def predict_dev():
        ema.eval();values=[]
        for d in dev_days:
            z=torch.from_numpy(np.array(x[d:d+1],np.float32)).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                p=ema(z,graph)[0].float().cpu().numpy()
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite development predictions')
            values.append(p)
        return np.stack(values)
    def metrics(p):
        sq=(p.astype(np.float64)-truth)**2
        groups=[]
        for regime in ['R1','R2','R3']:
            ids=[i for i,d in enumerate(dev_days) if pm['regimes'][d]==regime]
            m=target[ids];n=int(m.sum())
            mse=(sq[ids]*m[:,None]).sum((0,2,3))/n
            groups.append(np.sqrt(mse))
        macro=np.stack(groups).mean(0)
        pooled=np.sqrt((sq*target[:,None]).sum((0,2,3))/target.sum())
        return dict(score=float(1-np.dot([.54,.46],macro)),
                    macro=(macro*np.array([25.,600.])).tolist(),
                    pooled=(pooled*np.array([25.,600.])).tolist())
    raw=predict_dev()
    base=np.stack([np.asarray(x[d,:2],np.float64)*np.array([4.,2000/600])[:,None,None] for d in dev_days])
    previous=np.maximum(0,base+alpha[None]*(raw-base))
    np.savez_compressed(out/'previous-development.npz',prediction=previous.astype(np.float32),days=dev_days)
    initial=metrics(previous);save(out/'previous-development-metrics.json',initial)
    archived=json.loads((V1/'specialists'/panel/'residual-calibration.json').read_text())
    assert abs(initial['score']-archived['calibrated_development_score'])<2e-6, 'Previous inference parity mismatch'
    selected=dict(epoch=0,blend=[0.,0.],metrics=initial,checkpoint=None)
    selected_predictions=previous.astype(np.float32)
    best_flow=initial['pooled'][1]
    calibration=json.loads((V1/'specialists'/panel/'batch-calibration.json').read_text())
    batch=calibration['selected']
    # Reuse proven memory sizes. No dummy allocations or unproductive VRAM filling.
    accumulate=2 if batch<=2 else 1
    original_batch=batch;original_accumulate=accumulate
    while batch*pm['nodes']>704:
        batch//=2;accumulate*=2
    save(out/'capacity-plan.json',dict(original_batch=original_batch,original_accumulation=original_accumulate,
        micro_batch=batch,accumulation=accumulate,nodes=pm['nodes'],node_batch_limit=704,
        effective_batch_unchanged=batch*accumulate==original_batch*original_accumulate))
    attention=[p for n,p in model.named_parameters() if n.startswith('time_attention.')]
    backbone=[p for n,p in model.named_parameters() if not n.startswith('time_attention.')]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':3e-5},{'params':attention,'lr':6e-5}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,growth_interval=2000)
    rng=np.random.default_rng(20261024)
    fit={r:[d for d in range(28,196) if pm['regimes'][d]==r] for r in ['R1','R2','R3']}
    def sample():
        r=['R1','R2','R3'][int(rng.integers(3))]
        days=rng.choice(fit[r],size=batch)
        return [np.array(a[days],np.float32) for a in [x,y,mask]]
    history=[dict(epoch=0,**initial)]
    pool=ThreadPoolExecutor(max_workers=1)
    steps=20;epochs=3;updates=0
    progress('initialized',micro_batch=batch,accumulation=accumulate,parameters=sum(p.numel() for p in model.parameters()),initial=initial)
    for epoch in range(1,epochs+1):
        model.train();running=0.;pending=pool.submit(sample)
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for j in range(accumulate):
                arrays=pending.result()
                if step<steps-1 or j<accumulate-1:pending=pool.submit(sample)
                features,labels,m=[torch.from_numpy(a).cuda() for a in arrays]
                with torch.autocast('cuda',dtype=torch.float16):
                    p=model(features,graph)
                    mse=(((p-labels).float()**2)*m[:,None]).sum((0,2,3))/m.sum().clamp_min(1)
                    loss=.35*(mse[0]+1e-8).sqrt()+.65*(mse[1]+1e-8).sqrt()+3*mse[1]+.05*mse[0]
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss/accumulate).backward();running+=float(loss.detach())/accumulate
                del arrays,features,labels,m,p,loss,mse
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();updates+=1
            factor=.15+.85*(1+math.cos(math.pi*updates/(epochs*steps)))/2
            optimizer.param_groups[0]['lr']=3e-5*factor;optimizer.param_groups[1]['lr']=6e-5*factor
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.1)
            if step%5==0:progress('training',epoch=epoch,step=step+1,steps=steps,loss=running/(step+1),gradient_norm=float(norm))
        predictions=predict_dev()
        # Channel-wise conservative mixing. Both pooled and regime-macro errors
        # must improve on internal development. Zero is always the fallback.
        blends=[]
        for channel in range(2):
            choices=[]
            for a in [0.,.25,.5,.75,1.]:
                value=previous.copy();value[:,channel]=(1-a)*previous[:,channel]+a*predictions[:,channel]
                mm=metrics(value)
                if mm['pooled'][channel]<=initial['pooled'][channel]+1e-9 and mm['macro'][channel]<=initial['macro'][channel]+1e-9:
                    choices.append((mm['pooled'][channel],a))
            blends.append(min(choices)[1])
        blend=np.array(blends)[None,:,None,None]
        mixed=(1-blend)*previous+blend*predictions
        result=metrics(mixed)
        accepted=result['pooled'][1]<best_flow-.02 and result['score']>=initial['score']+1e-5
        if accepted:
            best_flow=result['pooled'][1]
            selected=dict(epoch=epoch,blend=blends,metrics=result,checkpoint='best-model.pt')
            selected_predictions=mixed.astype(np.float32)
            torch.save(dict(state_dict=ema.state_dict(),config=config,epoch=epoch,development_score=result['score']),out/'best-model.pt')
        history.append(dict(epoch=epoch,blend=blends,metrics=result,accepted=accepted,loss=running/steps))
        save(out/'training-history.json',history)
        progress('epoch_completed',epoch=epoch,selected=selected,candidate=result,blend=blends)
    pool.shutdown(wait=True)
    np.savez_compressed(out/'selected-development.npz',prediction=selected_predictions,days=dev_days)
    selected.update(panel=panel,previous_route=route,micro_batch=batch,accumulation=accumulate,
                    selection='Internal dates 196:218 only; no holdout opened; both channels require nonworsening pooled and macro RMSE',
                    seconds=time.monotonic()-began)
    save(out/'selection.json',selected)
    progress('complete',selection=selected)


def controller():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'policy.json').exists():
        routes=json.loads((V1/'final-model.json').read_text())['routes']
        preserve=[V1/'final-model.json',V1/'candidate-submission-experts/submission.csv']
        preserve += [V1/r[key] for r in routes for key in ['checkpoint','calibration','model_graph']]
        save(ROOT/'preserved-v1-sha256.json',{str(p.resolve()):sha(p) for p in preserve})
        save(ROOT/'policy.json',dict(recorded_at=dt.datetime.now().astimezone().isoformat(),
            scope='Target three regressed corridors first; reduce flow RMSE on all ten',
            earlier_holdout_seen=True,holdout_labels_in_fit=False,fit_indices=[28,196],development_indices=[196,218],
            holdout_indices=[218,273],epochs=3,steps=20,seed=20261024,
            loss='.35 speed_RMSE + .65 flow_RMSE + 3 flow_MSE + .05 speed_MSE; normalized by 25 and 600',
            selection='One checkpoint per corridor chosen for lower internal pooled flow RMSE, >=0.02 physical unit gain and no score decrease; each channel mixing grid [0,.25,.5,.75,1] requires both pooled and macro RMSE not worsen.',
            holdout='Only after all ten routes are frozen. No choosing candidates using holdout.',
            gpu='Reuse previous proven actual batches, FP16, activation checkpointing, full 288-step corridors. No decorative memory allocations.',
            source_sha256={str(p):sha(p) for p in [Path(__file__),V1/'model.py',V1/'hybrid_model.py']}))
    priority=['D12_I5_N','D7_I405_N','D7_I210_E']
    order=priority+[r['panel'] for r in json.loads((V1/'final-model.json').read_text())['routes'] if r['panel'] not in priority]
    started=time.monotonic();rows=[]
    with (ROOT/'telemetry.csv').open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['elapsed_seconds']+['timestamp','ac_connected']+FIELDS)
        if f.tell()==0:writer.writeheader()
        for i,panel in enumerate(order):
            out=ROOT/panel;out.mkdir(exist_ok=True)
            save(ROOT/'progress.json',dict(phase='training',panel=panel,index=i+1,total=10))
            if (out/'selection.json').exists():continue
            with (out/'training.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__)), '--worker',panel],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                reason=None
                try:
                    while child.poll() is None:
                        row=telemetry();row['elapsed_seconds']=time.monotonic()-started
                        rows.append(row);writer.writerow(row);f.flush()
                        if row['temperature.gpu']>=86:reason='Temperature limit'
                        if not row['ac_connected']:reason='AC disconnected'
                        if (ROOT/'STOP').exists():reason='Stop requested'
                        if reason:raise RuntimeError(reason)
                        time.sleep(3)
                finally:
                    if child.poll() is None:child.terminate();child.wait(timeout=20)
                if child.returncode:raise RuntimeError('Worker failed: '+panel)
    selected=[json.loads((ROOT/p/'selection.json').read_text()) for p in order]
    save(ROOT/'frozen-routes.json',dict(routes=selected,frozen_at=dt.datetime.now().astimezone().isoformat(),holdout_used=False))
    with (ROOT/'telemetry.csv').open(newline='',encoding='utf-8') as f:recorded=list(csv.DictReader(f))
    save(ROOT/'monitor-summary.json',dict(seconds=sum(r['seconds'] for r in selected),
        current_controller_seconds=time.monotonic()-started,includes_failed_attempt_peak=True,
        peak_vram_mib=max(float(r['memory.used']) for r in recorded),peak_power_w=max(float(r['power.draw']) for r in recorded),
        peak_temperature_c=max(float(r['temperature.gpu']) for r in recorded)))
    save(ROOT/'progress.json',dict(phase='routes_frozen',completed=10,total=10))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    try:worker(args.worker) if args.worker else controller()
    except Exception:
        traceback.print_exc();sys.exit(1)
