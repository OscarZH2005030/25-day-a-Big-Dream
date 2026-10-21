"""Prepare/train correction heads, freeze all choices before holdout evaluation."""
import argparse
import copy
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from residual_core import *


def worker(panel):
    torch.set_num_threads(4);torch.manual_seed(20261026)
    torch.cuda.set_per_process_memory_fraction(.89)
    began=time.monotonic();out=ROOT/panel
    info=json.loads((out/'prepared.json').read_text());source=Path(info['source'])
    pm=next(p for p in json.loads((CACHE/'metadata.json').read_text())['panels'] if p['panel']==panel)
    model=ResidualFlow(pm['nodes']).cuda();ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    fitx=torch.from_numpy(np.load(out/'fit-features.npy')).cuda()
    fitbase=torch.from_numpy(np.load(out/'fit-base.npy')).cuda()
    fity=torch.from_numpy(np.load(source/'fit-y.npy').astype(np.float32)).cuda()
    fitnode=torch.from_numpy(np.load(source/'fit-nodes.npy').astype(np.int64)).cuda()
    dx=np.load(out/'dev-features.npy',mmap_mode='r');base=np.load(out/'dev-base.npy')
    truth=np.load(source/'dev-y.npy').astype(np.float64);nodes=np.load(source/'dev-nodes.npy');days=np.load(source/'dev-days.npy')
    regimes=np.array([pm['regimes'][d] for d in days])
    groups=[np.ones(len(days),bool),days<207,days>=207]
    reference=[flow_metrics(base,truth,days,regimes,g) for g in groups]
    expected=json.loads((BASE/'final-model.json').read_text())
    old=next(r for r in expected['routes'] if r['panel']==panel)
    local=next(r for r in expected['local_flow_routes'] if r['panel']==panel)
    expected_flow=local['pooled_flow'] if local['alpha'] else old['metrics']['pooled'][1]
    assert abs(reference[0][0]-expected_flow)<.005, 'Frozen v2 baseline mismatch'
    # This identity check covers real fitting examples and prevents low precision
    # feature storage from changing the baseline before learning any correction.
    ids=torch.arange(min(1024,len(fity)),device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):initial=ema(fitx[ids].float(),fitnode[ids],fitbase[ids])
    assert torch.equal(initial,fitbase[ids])
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    def loss_fn(p,labels,b):return (p-labels).square().mean()+.05*(p-b).square().mean()
    batch=32768;trials=[]
    for candidate in [32768,65536,131072,262144,524288]:
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        ids=p=loss=None
        try:
            ids=torch.randint(len(fity),(candidate,),device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                p=model(fitx[ids].float(),fitnode[ids],fitbase[ids]);loss=loss_fn(p,fity[ids],fitbase[ids])
            loss.backward();torch.cuda.synchronize()
            allocated=torch.cuda.max_memory_allocated()/1024**3
            trials.append(dict(batch=candidate,allocated_gib=allocated,status='fits'))
            if allocated>10.5:
                trials[-1]['status']='reserve_training_headroom';break
            batch=candidate
        except torch.OutOfMemoryError:
            trials.append(dict(batch=candidate,status='allocator_limit'));break
        finally:
            ids=p=loss=None;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    save(out/'batch-plan.json',dict(batch=batch,trials=trials))
    best=reference[0][0]
    choice=dict(panel=panel,alpha=0.,step=0,pooled_flow=float(best),macro_flow=float(reference[0][1]),batch=batch,
        parameters=sum(p.numel() for p in model.parameters()),initial_identity_verified=True)
    history=[]
    for step in range(1,501):
        model.train();optimizer.zero_grad(set_to_none=True)
        ids=torch.randint(len(fity),(batch,),device='cuda')
        with torch.autocast('cuda',dtype=torch.float16):
            p=model(fitx[ids].float(),fitnode[ids],fitbase[ids]);loss=loss_fn(p,fity[ids],fitbase[ids])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        for group in optimizer.param_groups:group['lr']=2e-4*(.1+.9*(1+np.cos(np.pi*step/500))/2)
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.05)
        if step%25==0:
            progress=dict(phase='training',panel=panel,step=step,steps=500,batch=batch,loss=float(loss.detach()),seconds=time.monotonic()-began)
            save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        if step in [100,300,500]:
            raw=infer_rows(ema,dx,nodes,base);candidates=[]
            for alpha in [0.,.1,.25,.5,.75,1.]:
                pred=(1-alpha)*base.astype(np.float64)+alpha*raw
                metrics=[flow_metrics(pred,truth,days,regimes,g) for g in groups]
                if all(np.all(a<=b+1e-9) for a,b in zip(metrics,reference)):
                    candidates.append((metrics[0][0],alpha,metrics[0][1]))
            score,alpha,macro=min(candidates)
            if score<best-.01:
                best=score;choice.update(alpha=alpha,step=step,pooled_flow=float(score),macro_flow=float(macro))
                torch.save(dict(state_dict=ema.state_dict(),nodes=pm['nodes'],features=FEATURES),out/'best-model.pt')
                np.save(out/'selected-dev-flow.npy',((1-alpha)*base+alpha*raw).astype(np.float32))
            history.append(dict(step=step,raw_pooled_flow=float(flow_metrics(raw,truth,days,regimes,groups[0])[0]),selected=choice.copy()))
            save(out/'history.json',history)
    choice.update(seconds=time.monotonic()-began,holdout_used=False)
    save(out/'selection.json',choice);print(json.dumps(choice),flush=True)


def controller():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/'policy.json').exists():
        manifest=json.loads((BASE/'final-model.json').read_text());files=[BASE/'final-model.json',BASE/'candidate-submission/submission.csv']
        files += [BASE/r['panel']/r['checkpoint'] for r in manifest['routes'] if r['checkpoint']]
        files += [BASE/'local-flow'/r['panel']/name for r in manifest['local_flow_routes'] if r['alpha'] for name in ['best-model.pt','fit-profiles.npz']]
        save(ROOT/'preserved-v2-sha256.json',{str(p.resolve()):sha(p) for p in files})
        save(ROOT/'policy.json',dict(recorded_at=dt.datetime.now().astimezone().isoformat(),
            reference='Current v2 model only; original Transformer is unavailable and not compared.',
            baseline_score=.9397804646468753,baseline_flow_rmse=34.6501080550362,
            panels=PANELS,fit_indices=[28,196],dev_indices=[196,218],holdout_indices=[218,273],earlier_holdout_seen=True,
            architecture='Frozen v2 model + zero-initialized 4-block width512 flow-error correction network, node embedding32, input136',
            steps=500,evaluation_steps=[100,300,500],seed=20261026,loss='Flow MSE + .05 correction squared; float32 baseline preserves identity',
            selection='Mixing [0,.1,.25,.5,.75,1], pooled and regime-macro errors must not worsen in both early and late development halves and full development. Minimum pooled gain0.01. Freeze all selections before evaluation.',
            limitation='Residual training uses in-sample predictions of a frozen supervised base; development remains outside both fitting periods.',
            source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    rows=[];began=time.monotonic()
    with (ROOT/'telemetry.csv').open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['timestamp','elapsed_seconds','ac_connected']+FIELDS)
        if f.tell()==0:writer.writeheader()
        for i,panel in enumerate(PANELS):
            out=ROOT/panel;out.mkdir(exist_ok=True)
            for phase in ['prepare','worker']:
                done=out/('prepared.json' if phase=='prepare' else 'selection.json')
                if done.exists():continue
                save(ROOT/'progress.json',dict(phase=phase,panel=panel,index=i+1,total=3))
                with (out/(phase+'.log')).open('w',encoding='utf-8') as log:
                    child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__)),'--'+phase,panel],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        while child.poll() is None:
                            row=telemetry();row['elapsed_seconds']=time.monotonic()-began;rows.append(row)
                            writer.writerow(row);f.flush()
                            if row['temperature.gpu']>=86 or not row['ac_connected'] or (ROOT/'STOP').exists():raise RuntimeError('Resource watchdog stop')
                            time.sleep(3)
                    finally:
                        if child.poll() is None:child.terminate();child.wait(timeout=20)
                    if child.returncode:raise RuntimeError('Stage failed: '+panel+'/'+phase)
    choices=[json.loads((ROOT/p/'selection.json').read_text()) for p in PANELS]
    save(ROOT/'frozen-routes.json',dict(baseline=str(BASE/'final-model.json'),frozen_at=dt.datetime.now().astimezone().isoformat(),holdout_used=False,routes=choices,
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    with (ROOT/'telemetry.csv').open(newline='',encoding='utf-8') as f:recorded=list(csv.DictReader(f))
    save(ROOT/'monitor-summary.json',dict(seconds=time.monotonic()-began,peak_vram_mib=max(float(r['memory.used']) for r in recorded),
        peak_power_w=max(float(r['power.draw']) for r in recorded),peak_temperature_c=max(float(r['temperature.gpu']) for r in recorded)))
    save(ROOT/'progress.json',dict(phase='routes_frozen',adopted=sum(c['alpha']>0 for c in choices)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare');p.add_argument('--worker');a=p.parse_args()
    try:
        if a.prepare:prepare(a.prepare)
        elif a.worker:worker(a.worker)
        else:controller()
    except Exception:traceback.print_exc();sys.exit(1)
