"""Train a larger graph-SSM on the exact team time80/20 date/target protocol."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from gpu_stress_test import telemetry,FIELDS,save


def worker(args):
    import numpy as np
    import pandas as pd
    import torch
    from model import GapGraphSSM
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(.89)
    torch.backends.cudnn.benchmark=False
    rng=np.random.default_rng(args.seed)
    began=time.monotonic()
    out=args.out
    def progress(phase,**details):
        data=dict(phase=phase,seconds=time.monotonic()-began)
        data.update(details);save(out/'progress.json',data);print(json.dumps(data),flush=True)
    metadata=json.loads((args.cache/'metadata.json').read_text())
    panels=[];offset=0
    for pm in metadata['panels']:
        root=args.cache/pm['panel'];old=Path(pm['old_cache'])
        graph={k:torch.from_numpy(v).cuda() for k,v in np.load(root/'graph.npz').items()}
        graph['node_id']=torch.arange(offset,offset+pm['nodes'],device='cuda');offset+=pm['nodes']
        panels.append(dict(meta=pm,x=np.load(root/'features.npy',mmap_mode='r'),
                           y=np.load(old/'y.npy',mmap_mode='r'),mask=np.load(old/'target.npy',mmap_mode='r'),graph=graph,
                           fit={r:[i for i in range(28,196) if pm['regimes'][i]==r] for r in ['R1','R2','R3']}))
    assert offset<=2048
    model=GapGraphSSM(len(metadata['features']),args.width,args.depth).cuda()
    model.load_state_dict(torch.load(args.init_checkpoint,map_location="cuda",weights_only=True)["state_dict"])
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024,growth_interval=2000)
    total_updates=args.epochs*args.steps
    def lr_factor(step):
        if step<args.steps:return .15+.85*step/args.steps
        fraction=(step-args.steps)/max(1,total_updates-args.steps)
        import math
        return .08+.92*(1+math.cos(math.pi*fraction))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    weights=torch.tensor([.54,.46],device='cuda')
    config=dict(architecture='gap-aware multi-hop graph + bidirectional real diagonal SSM',
                parameters=parameters,width=args.width,depth=args.depth,features=metadata['features'],
                temporal_window=288,spatial_context='entire corridor',seed=args.seed,
                train_dates=metadata['fit_dates'],profile_dates=metadata['history_dates'],
                development_dates=metadata['eval_dates'],holdout_dates=metadata['holdout_dates'],
                original_transformer_reference=0.9358124847434317,
                note='Shared model across ten corridors; all template training targets; no Transformer attention.')
    config['warm_start']=str(args.init_checkpoint.resolve())
    config['phase']='low-learning-rate refinement from primary internal-development winner'
    save(out/'config.json',config)
    def host_batch(p,count,regime=None):
        days=rng.choice(p['fit'][regime] if regime else np.arange(28,196),size=count)
        return (np.array(p['x'][days],dtype=np.float32),np.array(p['y'][days],dtype=np.float32),
                np.array(p['mask'][days],dtype=np.float32))
    def to_device(arrays):
        x,y,m=[torch.from_numpy(a).cuda() for a in arrays]
        return x,y,m
    def loss_fn(pred,y,mask):
        squared=((pred-y).float()**2)*mask[:,None]
        mse=squared.sum((0,2,3))/mask.sum().clamp_min(1)
        return ((mse+1e-8).sqrt()*weights).sum()+.05*(mse*weights).sum()
    # Calibrate with the largest real corridor and a fitting-period day.
    worst=max(panels,key=lambda p:p['meta']['nodes'])
    selected=1;trials=[]
    for candidate in [1,2,3,4]:
        x=y=mask=pred=loss=None
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        progress('batch_probe',candidate=candidate,panel=worst['meta']['panel'])
        try:
            x,y,mask=to_device(host_batch(worst,candidate))
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(x,worst['graph']);loss=loss_fn(pred,y,mask)
            loss.backward();torch.cuda.synchronize()
            allocated=torch.cuda.max_memory_allocated()/1024**3;reserved=torch.cuda.max_memory_reserved()/1024**3
            trials.append(dict(batch=candidate,allocated_gib=allocated,reserved_gib=reserved,status='fits'))
            selected=candidate
            if reserved>=13.:break
        except torch.OutOfMemoryError:
            trials.append(dict(batch=candidate,status='allocator_limit'));break
        finally:
            x=y=mask=pred=loss=None;optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    args.batch=selected
    save(out/'batch-calibration.json',dict(selected=selected,trials=trials,gradient_accumulation=args.accumulate))
    progress('calibrated',micro_batch=selected,effective_batch=selected*args.accumulate,parameters=parameters)
    ema=copy.deepcopy(model).eval()
    for parameter in ema.parameters():parameter.requires_grad_(False)
    def evaluate(days,name,save_predictions=False):
        ema.eval();rows=[];total_sq=np.zeros(2);initial_sq=np.zeros(2);count=0
        for p in panels:
            sums={r:np.zeros(5,dtype=np.float64) for r in ['R1','R2','R3']}
            chunks=[]
            for d in days:
                x=torch.from_numpy(np.array(p['x'][d:d+1],dtype=np.float32)).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    pred=ema(x,p['graph'])[0].cpu().numpy()
                if not np.isfinite(pred).all():raise RuntimeError('Nonfinite prediction')
                mask=p['mask'][d];truth=p['y'][d]
                initial=np.asarray(p['x'][d,:2],dtype=np.float32)*np.array([4.,2000/600])[:,None,None]
                err=(pred-truth)[:,mask].astype(np.float64)
                base=(initial-truth)[:,mask].astype(np.float64)
                sq=(err**2).sum(1);bsq=(base**2).sum(1);n=int(mask.sum())
                sums[p['meta']['regimes'][d]]+=[n,*sq,*bsq]
                total_sq+=sq;initial_sq+=bsq;count+=n
                if save_predictions:chunks.append(pred.astype(np.float32))
            for regime,s in sums.items():
                n=s[0];rmse=np.sqrt(s[1:3]/n);baseline=np.sqrt(s[3:5]/n)
                score=float(np.dot([.54,.46],np.maximum(0,1-rmse)))
                bscore=float(np.dot([.54,.46],np.maximum(0,1-baseline)))
                rows.append(dict(panel=p['meta']['panel'],family=p['meta']['family'],regime=regime,n=int(n),
                                 rmse_speed_kmh=float(rmse[0]*25),rmse_flow_per_lane=float(rmse[1]*600),
                                 S_state=score,initial_S_state=bscore))
            if save_predictions:
                np.savez_compressed(out/(p['meta']['panel']+'-holdout.npz'),prediction=np.stack(chunks),days=np.array(list(days)))
            progress('evaluation',split=name,panel=p['meta']['panel'])
        table=pd.DataFrame(rows)
        score=float(table.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean())
        baseline=float(table.groupby(['family','panel']).initial_S_state.mean().groupby('family').mean().mean())
        table.to_csv(out/(name+'-by-regime.csv'),index=False)
        result=dict(split=name,score=score,initial_score=baseline,n=count,
                    pooled_speed_rmse=float(np.sqrt(total_sq[0]/count)*25),
                    pooled_flow_rmse_per_lane=float(np.sqrt(total_sq[1]/count)*600))
        save(out/(name+'-metrics.json'),result)
        return result
    baseline=evaluate(range(196,218),'initial-development')
    best=baseline['score'];best_epoch=0;stale=0;total_targets=0
    history=[dict(epoch=0,training_loss=None,seconds=time.monotonic()-began,development_score=best,initial_score=baseline['initial_score'])]
    torch.save(dict(state_dict=ema.state_dict(),config=config,epoch=0,development_score=best),out/'best-model.pt')
    executor=ThreadPoolExecutor(max_workers=1)
    def make_sample():
        p=panels[int(rng.integers(len(panels)))];regime=['R1','R2','R3'][int(rng.integers(3))]
        return p,host_batch(p,args.batch,regime)
    # Source hashes and protocol are recorded alongside the weights.
    source_hash={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    save(out/'source-hashes.json',source_hash)
    for epoch in range(1,args.epochs+1):
        model.train();running=0.;updates=0
        pending=executor.submit(make_sample)
        for step in range(1,args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            for accumulation in range(args.accumulate):
                p,arrays=pending.result()
                final=step==args.steps and accumulation==args.accumulate-1
                if not final:pending=executor.submit(make_sample)
                x,y,mask=to_device(arrays)
                with torch.autocast('cuda',dtype=torch.float16):
                    pred=model(x,p['graph']);loss=loss_fn(pred,y,mask)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                scaler.scale(loss/args.accumulate).backward()
                running+=float(loss.detach())/args.accumulate
                total_targets+=int(mask.sum().item())
                del x,y,mask,pred,loss,arrays
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();scheduler.step();updates+=1
            with torch.no_grad():
                for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.02)
            if step%10==0 or step==1:
                progress('training',epoch=epoch,epochs=args.epochs,step=step,steps=args.steps,
                         mean_loss=running/step,gradient_norm=float(norm),learning_rate=scheduler.get_last_lr()[0])
        record=dict(epoch=epoch,training_loss=running/updates,seconds=time.monotonic()-began)
        torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
                        rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),config=config),out/'last-training.pt')
        if epoch%args.eval_every==0 or epoch==args.epochs:
            metric=evaluate(range(196,218),'development')
            record.update(development_score=metric['score'],initial_score=metric['initial_score'])
            if metric['score']>best+1e-5:
                best=metric['score'];best_epoch=epoch;stale=0
                torch.save(dict(state_dict=ema.state_dict(),config=config,epoch=epoch,
                                development_score=best),out/'best-model.pt')
            else:stale+=1
        history.append(record);save(out/'training-history.json',history)
        progress('epoch_completed',**record,best_epoch=best_epoch,best_development_score=best)
        if stale>=args.patience:
            progress('early_stop',patience=args.patience);break
    executor.shutdown(wait=True)
    checkpoint=torch.load(out/'best-model.pt',map_location='cuda',weights_only=True)
    ema.load_state_dict(checkpoint['state_dict'])
    # The holdout is used exactly once, after checkpoint selection on internal development.
    held=evaluate(range(218,273),'holdout',save_predictions=True)
    reference=pd.read_csv(args.reference)
    current=pd.read_csv(out/'holdout-by-regime.csv')
    reference=reference[reference.regime.isin(['R1','R2','R3'])]
    joined=current.merge(reference[['panel','regime','n','S_state']],on=['panel','regime'],suffixes=('_new','_transformer'),validate='one_to_one')
    assert (joined.n_new==joined.n_transformer).all() and len(joined)==30
    joined['delta']=joined.S_state_new-joined.S_state_transformer
    joined.to_csv(out/'comparison-with-team-transformer.csv',index=False)
    reference_score=float(joined.groupby('panel').S_state_transformer.mean().mean())
    result=dict(status='completed',architecture=config['architecture'],parameters=parameters,
                trained_epochs=epoch,micro_batch=args.batch,gradient_accumulation=args.accumulate,
                best_epoch=best_epoch,best_development_score=best,holdout=held,
                team_transformer_same_holdout_score=reference_score,delta_holdout=held['score']-reference_score,
                targets_seen_including_repeats=total_targets,seconds=time.monotonic()-began,
                protocol='Exact date split and complete holdout target counts matched; shared cross-corridor model versus team per-corridor models.',
                limitation='Archived Transformer predictions/weights are not provided; no paired daily uncertainty estimate or retrained reference in this run.')
    save(out/'training-result.json',result);progress('complete',**result)


def monitor(args):
    args.out.mkdir(parents=True,exist_ok=True)
    initial=telemetry()
    if not initial['ac_connected'] or initial['temperature.gpu']>=86:raise RuntimeError('AC power required and temperature must be below 86 C')
    start=time.monotonic();started=dt.datetime.now().astimezone().isoformat();rows=[];reason=None;failed=0
    with (args.out/'training.log').open('w',encoding='utf-8') as log,(args.out/'telemetry.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['timestamp','elapsed_seconds','ac_connected']+FIELDS);writer.writeheader()
        command=[sys.executable,'-X','utf8',str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while child.poll() is None:
                try:
                    row=telemetry();row['elapsed_seconds']=time.monotonic()-start;rows.append(row)
                    writer.writerow(row);f.flush();failed=0
                    if row['temperature.gpu']>=86:reason='Temperature limit'
                    if not row['ac_connected']:reason='AC disconnected'
                except Exception as exc:
                    failed+=1
                    if failed>=3:reason=str(exc)
                if (args.out/'STOP').exists():reason='User stop requested'
                if time.monotonic()-start>7200:reason='Two-hour bounded training limit'
                if reason:child.terminate();child.wait(timeout=20);break
                time.sleep(2)
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=20)
    save(args.out/'monitor-summary.json',dict(status='completed' if child.returncode==0 and not reason else 'stopped',
        child_exit_code=child.returncode,stop_reason=reason,started=started,ended=dt.datetime.now().astimezone().isoformat(),
        seconds=time.monotonic()-start,peak_vram_mib=max(r['memory.used'] for r in rows),
        peak_temperature_c=max(r['temperature.gpu'] for r in rows),peak_power_w=max(r['power.draw'] for r in rows)))
    if child.returncode:raise RuntimeError('Training failed; inspect training.log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--steps',type=int,default=60);p.add_argument('--accumulate',type=int,default=2)
    p.add_argument('--width',type=int,default=256);p.add_argument('--depth',type=int,default=10)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--eval-every',type=int,default=2);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--init-checkpoint',type=Path,required=True)
    p.add_argument('--worker',action='store_true');args=p.parse_args()
    try:worker(args) if args.worker else monitor(args)
    except Exception:traceback.print_exc();sys.exit(1)
