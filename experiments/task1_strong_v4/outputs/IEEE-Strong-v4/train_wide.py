"""Expand the principal network 3.85x; keep all existing v3 corrections."""
import argparse
import copy
import csv
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from common import *
from wide_model import verify_cpu


def worker(panel):
    torch.set_num_threads(4);torch.manual_seed(20261027)
    torch.cuda.set_per_process_memory_fraction(.90)
    out=ROOT/panel;out.mkdir(exist_ok=True);start=time.monotonic()
    def progress(phase,**kw):
        data=dict(panel=panel,phase=phase,seconds=time.monotonic()-start,**kw)
        save(out/'progress.json',data);print(json.dumps(data),flush=True)
    pm=metadata(panel);old=Path(pm['old_cache'])
    x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
    y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
    narrow,graph=narrow_model(pm)
    for p in narrow.parameters():p.requires_grad_(False)
    model=WideGraphSSM().cuda();model.load_state_dict(expand_state(narrow.state_dict()))
    model.eval()
    # Float32 identity tests the trained model on actual fitting inputs.
    z=torch.from_numpy(np.array(x[28:29],np.float32)).cuda()
    with torch.inference_mode():fp32diff=float((model(z,graph)-narrow(z,graph)).abs().max())
    assert fp32diff<3e-5,fp32diff
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        amp_diff=float((model(z,graph)-narrow(z,graph)).abs().max())
    save(out/'expansion-parity.json',dict(passed=True,float32_maximum_difference=fp32diff,
        fp16_maximum_difference=amp_diff,features=64,width=512,depth=10,heads=16,steps_per_day=288,nodes=pm['nodes']))
    del z;torch.cuda.empty_cache()
    current=current_dev(pm,mask);target=np.array(mask[196:218]);truth=np.array(y[196:218,1],np.float64)
    regimes=np.array(pm['regimes'][196:218]);parts=[np.ones(22,bool),np.arange(22)<11,np.arange(22)>=11]
    reference=[flow_metrics(current[:,1],truth,target,regimes,p) for p in parts]
    expected=next(r for r in json.loads((V3/'frozen-routes.json').read_text())['routes'] if r['panel']==panel)
    assert abs(reference[0][0]-expected['pooled_flow'])<1e-5
    raw_dev=[]
    for d in range(196,218):raw_dev.append(predict(narrow,x[d],graph)[1])
    raw_dev=np.stack(raw_dev);np.save(out/'narrow-dev-flow.npy',raw_dev)
    progress('preparing_teacher',initial_pooled_flow=float(reference[0][0]))
    # Teacher uses the same leave-own-day-out training features as v3. No new
    # fitting, validation labels or holdout labels enter these predictions.
    source=V3/panel;local=BASE/'local-flow'/panel
    fitx=np.load(source/'fit-features.npy',mmap_mode='r');fitbase=np.load(source/'fit-base.npy',mmap_mode='r')
    nodes=np.load(local/'fit-nodes.npy');days=np.load(local/'fit-days.npy')
    expert=rc.ResidualFlow(pm['nodes']).cuda().eval()
    expert.load_state_dict(torch.load(source/'best-model.pt',map_location='cuda',weights_only=True)['state_dict'])
    teacher=rc.infer_rows(expert,fitx,nodes,fitbase);np.save(out/'teacher-fit-flow.npy',teacher)
    del expert,fitx,fitbase;torch.cuda.empty_cache()
    boundaries={d:np.flatnonzero(days==d) for d in range(28,196)}
    # EMA and optimizer state are present during the real batch feasibility test.
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512,growth_interval=2000)
    rng=np.random.default_rng(20261027)
    fit={r:[d for d in range(28,196) if pm['regimes'][d]==r] for r in ['R1','R2','R3']}
    def loss_for(selected_days):
        z=torch.from_numpy(np.array(x[selected_days],np.float32)).cuda()
        labels=torch.from_numpy(np.array(y[selected_days],np.float32)).cuda()
        m=torch.from_numpy(np.array(mask[selected_days])).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):oldp=narrow(z,graph).detach()
        teachers=torch.from_numpy(np.concatenate([teacher[boundaries[d]] for d in selected_days])).cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            p=model(z,graph)
            # Deployment is current_v3 + (new_wide - frozen_v2_raw). Train the
            # identical flow expression, including all existing v3 improvements.
            corrected=teachers+(p[:,1][m].float()-oldp[:,1][m].float())
            mse=(corrected-labels[:,1][m]).square().mean()
            consistency=(p[:,0][m].float()-oldp[:,0][m].float()).square().mean()
            return .65*(mse+1e-8).sqrt()+3*mse+.05*consistency
    batch=1;trials=[]
    for candidate in [1,2]:
        progress('capacity_probe',candidate=candidate)
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model.train()
        loss=None
        try:
            loss=loss_for([28]*candidate);scaler.scale(loss).backward()
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            # Exercise real Adam states; restore the exact warm start afterwards.
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            allocated=torch.cuda.max_memory_allocated()/1024**3
            reserved=torch.cuda.max_memory_reserved()/1024**3
            trials.append(dict(batch=candidate,status='fits',allocated_gib=allocated,reserved_gib=reserved));batch=candidate
            if reserved>12.0:break
        except torch.OutOfMemoryError:
            trials.append(dict(batch=candidate,status='allocator_limit'))
            if candidate==1:raise
            break
        finally:
            loss=None;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    del optimizer,scaler
    model.load_state_dict(expand_state(narrow.state_dict()));ema.load_state_dict(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=512,growth_interval=2000)
    accumulate=max(1,2//batch)
    save(out/'capacity-plan.json',dict(micro_batch=batch,accumulation=accumulate,effective_batch=batch*accumulate,trials=trials))
    best=float(reference[0][0]);selected=dict(panel=panel,alpha=0.,step=0,pooled_flow=best,macro_flow=float(reference[0][1]),
        checkpoint=None,parameters=sum(p.numel() for p in model.parameters()),micro_batch=batch,accumulation=accumulate)
    history=[];selected_prediction=current[:,1].copy()
    for step in range(1,61):
        model.train();optimizer.zero_grad(set_to_none=True);running=0.
        for _ in range(accumulate):
            regime=['R1','R2','R3'][int(rng.integers(3))];ds=rng.choice(fit[regime],size=batch).tolist()
            loss=loss_for(ds)
            assert torch.isfinite(loss)
            scaler.scale(loss/accumulate).backward();running+=float(loss.detach())/accumulate
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
        scaler.step(optimizer);scaler.update()
        for group in optimizer.param_groups:group['lr']=2e-5*(.15+.85*(1+math.cos(math.pi*step/60))/2)
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.1)
        if step%5==0:progress('training',step=step,total=60,loss=running,gradient_norm=float(norm))
        if step in [20,40,60]:
            raw=np.stack([predict(ema,x[d],graph)[1] for d in range(196,218)])
            choices=[]
            for alpha in [0.,.1,.25,.5,.75,1.]:
                mixed=np.maximum(0,current[:,1]+alpha*(raw-raw_dev))
                mm=[flow_metrics(mixed,truth,target,regimes,p) for p in parts]
                if all(np.all(a<=b+1e-8) for a,b in zip(mm,reference)):
                    choices.append((float(mm[0][0]),alpha,float(mm[0][1]),mixed))
            score,alpha,macro,value=min(choices,key=lambda z:z[0])
            accepted=score<best-.01
            if accepted:
                best=score;selected.update(alpha=alpha,step=step,pooled_flow=score,macro_flow=macro,checkpoint='best-model.pt')
                selected_prediction=value
                torch.save(dict(state_dict=ema.state_dict(),config=dict(width=512,depth=10,heads=16,features=64,variant='wide_graph_ssm'),step=step),out/'best-model.pt')
            history.append(dict(step=step,alpha=alpha,pooled_flow=score,macro_flow=macro,accepted=accepted))
            save(out/'history.json',history);progress('development_checked',**history[-1])
    np.save(out/'selected-dev-flow.npy',selected_prediction)
    selected.update(seconds=time.monotonic()-start,holdout_used=False)
    save(out/'selection.json',selected);progress('complete',selection=selected)


def controller():
    save(ROOT/'cpu-expansion-check.json',verify_cpu())
    preserve=json.loads((V3/'preserved-v2-sha256.json').read_text())
    for p in [V3/'frozen-routes.json',V3/'candidate-submission/submission.csv']+[V3/q/'best-model.pt' for q in PANELS]:preserve[str(p.resolve())]=sha(p)
    save(ROOT/'preserved-best-sha256.json',preserve)
    save(ROOT/'policy.json',dict(recorded_at=dt.datetime.now().astimezone().isoformat(),baseline='IEEE-Strong-v3',
        fit=[28,196],development=[196,218],holdout=[218,273],holdout_previously_seen=True,
        architecture='512 width,10 graph/SSM blocks,16 attention heads; 43,855,662 parameters,3.85x base',
        initialization='Duplicate channels and attention heads, split input coefficients, cancelling 1% perturbations; real FP32 parity checked.',
        training='60 updates per direction, EMA.1, AdamW2e-5 cosine, effective batch2; full288 steps/fullcorridor; FP16/activation checkpointing',
        deployment='Flow=current_v3 + alpha*(wide_model-frozen_v2_raw_model); speeds and other outputs unchanged.',
        selection='Check20,40,60; alpha[0,.1,.25,.5,.75,1]; pooled AND regime macro must not worsen on all/early11/late11 dev; minimum .01 pooled flow gain.',
        safeguards='Freeze all selections before local holdout. Never overwrite v3. If no accepted expansion, reuse v3.',
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    start=time.monotonic();rows=[]
    with (ROOT/'telemetry.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['elapsed_seconds','timestamp','ac_connected']+FIELDS);writer.writeheader()
        for i,panel in enumerate(PANELS):
            out=ROOT/panel;out.mkdir(exist_ok=True)
            save(ROOT/'progress.json',dict(phase='training',panel=panel,index=i+1,total=3))
            with (out/'worker.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__)),'--worker',panel],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    while child.poll() is None:
                        row=telemetry();row['elapsed_seconds']=time.monotonic()-start;rows.append(row);writer.writerow(row);f.flush()
                        if row['temperature.gpu']>=86 or not row['ac_connected'] or (ROOT/'STOP').exists():raise RuntimeError('Temperature, AC or stop guard triggered')
                        time.sleep(3)
                finally:
                    if child.poll() is None:child.terminate();child.wait(timeout=20)
                if child.returncode:raise RuntimeError('Worker failed: '+panel)
    routes=[json.loads((ROOT/p/'selection.json').read_text()) for p in PANELS]
    save(ROOT/'frozen-routes.json',dict(routes=routes,frozen_at=dt.datetime.now().astimezone().isoformat(),holdout_used=False,
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')}))
    save(ROOT/'monitor-summary.json',dict(seconds=time.monotonic()-start,peak_vram_mib=max(r['memory.used'] for r in rows),
        peak_power_w=max(r['power.draw'] for r in rows),peak_temperature_c=max(r['temperature.gpu'] for r in rows)))
    save(ROOT/'progress.json',dict(phase='routes_frozen',completed=3))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    try:worker(args.worker) if args.worker else controller()
    except Exception:traceback.print_exc();sys.exit(1)
