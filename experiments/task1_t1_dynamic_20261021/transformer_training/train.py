from __future__ import annotations
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import torch
from task1.cache import cache_key
from task1.data import save_json,sha256
from task1.transformer_priority.common import read,upstream
from task1.transformer_priority.model import model_for,objective
from task1.transformer_priority.train import Coverage,internal_score,predict,checkpoint_read
from task1.wave_neural.cache import save_checkpoint,save_history
from task1.wave_neural.cache import replace_with_retry
from .common import code_identity,AUGMENT


def relative_snapshot(folder,names):
    return [dict(name=name,sha256=sha256(folder/name)) for name in sorted(set(names))]


def write_pointer(path,payload):
    temporary=path.with_name(path.name+'.tmp_'+uuid.uuid4().hex)
    save_json(temporary,payload)
    replace_with_retry(temporary,path)


def verify_model(folder):
    complete=read(folder/'COMPLETE.json')
    for item in complete['artifacts']:
        path=(folder/item['name']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path)!=item['sha256']:
            raise ValueError('Completed new model artifact changed')
    return complete


class Exposure:
    def __init__(self,state=None):
        self.rows=state if state is not None else []
        self.current={r:dict(windows=0,supervised=0,original_available=0,retained=0,extra=0,
                              speed_sse=0.,flow_sse=0.,retained_speed_sse=0.,retained_flow_sse=0.,
                              extra_speed_sse=0.,extra_flow_sse=0.,point=0,temporal=0,spatial=0,point_fill_units=0)
                      for r in ['R1','R2','R3']}

    def update(self,pred,batch,infos):
        error=(pred.detach().cpu()-batch['label'].detach().cpu()).numpy()
        target=batch['target'].detach().cpu().numpy()
        original=batch['original_target'].detach().cpu().numpy()
        for i,info in enumerate(infos):
            row=self.current[info['regime']]
            row['windows']+=1;row['supervised']+=info['supervised_targets']
            row['original_available']+=info['original_targets'];row['retained']+=info['retained_targets'];row['extra']+=info['extra_targets']
            if info['pattern'] in row: row[info['pattern']]+=1
            row['point_fill_units']+=info.get('point_fill_units',0)
            squared=(error[i][target[i]]**2).sum(axis=0)
            row['speed_sse']+=float(squared[0]);row['flow_sse']+=float(squared[1])
            for label,mask in [('retained',target[i]&original[i]),('extra',target[i]&~original[i])]:
                squared=(error[i][mask]**2).sum(axis=0)
                row[label+'_speed_sse']+=float(squared[0]);row[label+'_flow_sse']+=float(squared[1])

    def finish_epoch(self,epoch):
        for regime,row in self.current.items():
            n=row['supervised']
            self.rows.append(dict(epoch=epoch,regime=regime,**row,
                speed_rmse_kmh=25*np.sqrt(row['speed_sse']/n) if n else None,
                flow_lane_rmse=600*np.sqrt(row['flow_sse']/n) if n else None,
                retained_speed_rmse_kmh=25*np.sqrt(row['retained_speed_sse']/row['retained']) if row['retained'] else None,
                retained_flow_lane_rmse=600*np.sqrt(row['retained_flow_sse']/row['retained']) if row['retained'] else None,
                extra_speed_rmse_kmh=25*np.sqrt(row['extra_speed_sse']/row['extra']) if row['extra'] else None,
                extra_flow_lane_rmse=600*np.sqrt(row['extra_flow_sse']/row['extra']) if row['extra'] else None))
        self.current={r:{k:0 for k in row} for r,row in self.current.items()}


def fit(data,sampler,config,variant,device,cache_root,source_identity,status,stop_after_epoch=None):
    for role in ['training','internal']:
        for date in data.roles[role]: data.labels(date,role)
    identity=dict(kind='t1_training_ablation_model_v1',panel=data.panel,variant=variant,config=config,
        features=[data.records[d]['key'] for d in data.roles['training']+data.roles['internal']],
        labels={d:r['key'] for d,r in data.label_records.items()},normalization=data.provenance['normalization_key'],
        roles={r:data.roles[r] for r in ['training','internal']},augmentation=AUGMENT if variant=='A_dynamic' else None,
        regime_sampling='one uniform regime per step, two within-regime windows' if variant=='B_regime' else 'original T1 positions/dates',
        implementation=code_identity(),source=source_identity)
    key=cache_key(identity);folder=Path(cache_root)/'models'/key;folder.mkdir(parents=True,exist_ok=True)
    model,initial_hash=model_for(config,'T1_balanced')
    if source_identity.get('expected_initial_sha256',initial_hash)!=initial_hash:
        raise ValueError('Initialization differs from original T1')
    with upstream().lock(folder/'worker.lock'):
        if (folder/'identity.json').exists() and read(folder/'identity.json')!=identity:
            raise ValueError('Training identity changed')
        save_json(folder/'identity.json',identity)
        if (folder/'COMPLETE.json').exists():
            complete=verify_model(folder)
            if complete['identity']!=key: raise ValueError('Completed model identity differs')
            checkpoint,best=checkpoint_read(folder,'best.json',key)
            model.load_state_dict(checkpoint['state_dict'])
            print('[new model HIT]',data.panel,variant,key[:12],flush=True)
            return model.to(device),dict(key=key,folder=str(folder),hit=True,best=best,selection=read(folder/'selection.json'))
        model=model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs']*config['steps_per_epoch'],eta_min=config['learning_rate']*.1)
        rng=np.random.default_rng(config['seed']);aux_rng=np.random.default_rng(config['seed']+811)
        coverage=Coverage(data);exposure=Exposure()
        best=-float('inf');stale=0;start_epoch=0;history=[];best_epoch=None;best_coverage=None
        if (folder/'last.json').exists():
            saved,_=checkpoint_read(folder,'last.json',key)
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            rng.bit_generator.state=saved['numpy_rng'];aux_rng.bit_generator.state=saved['augmentation_rng']
            torch.set_rng_state(saved['torch_rng'])
            if device.type=='cuda': torch.cuda.set_rng_state_all(saved['cuda_rng'])
            coverage=Coverage(data,saved['coverage']);exposure=Exposure(saved['exposure'])
            start_epoch=saved['epoch'];best=saved['best'];stale=saved['stale'];history=saved['history']
            best_epoch=saved['best_epoch'];best_coverage=saved['best_coverage']
            print('[new model resume]',data.panel,variant,start_epoch,flush=True)
        else:
            torch.manual_seed(config['seed']+123)
            if device.type=='cuda': torch.cuda.manual_seed_all(config['seed']+123)
        for epoch in range(start_epoch,config['epochs']):
            if stale>=config['patience']: break
            started=time.perf_counter();model.train();losses=[]
            for step in range(config['steps_per_epoch']):
                batch,graph,spec,infos=sampler.batch(rng,aux_rng,config)
                coverage.update(dict(batch,target=batch['target']&batch['original_target']),spec)
                batch={k:v.to(device) for k,v in batch.items()};graph={k:v.to(device) for k,v in graph.items()}
                optimizer.zero_grad(set_to_none=True)
                pred=model(batch,graph);loss=objective(pred,batch)
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
                optimizer.step();scheduler.step();losses.append(float(loss.detach()))
                exposure.update(pred,batch,infos)
                if step==0 or (step+1)%128==0:
                    status(stage='training',panel=data.panel,variant=variant,epoch=epoch+1,step=step+1)
                    print(f'[{data.panel} {variant}] epoch={epoch+1} step={step+1}/{config["steps_per_epoch"]} loss={np.mean(losses):.6f}',flush=True)
            status(stage='internal_validation',panel=data.panel,variant=variant,epoch=epoch+1)
            value=internal_score(model,data,config,device);improved=value>best
            stale=0 if improved else stale+1
            summary=coverage.summary()
            if improved: best=value;best_epoch=epoch+1;best_coverage=summary
            exposure.finish_epoch(epoch+1)
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),internal_S_state=value,best_S_state=best,
                seconds=time.perf_counter()-started,sampling_sha256=summary['sampling_sha256'],
                unique_original_targets_seen=summary['unique_targets_seen'])
            history.append(row)
            saved=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                epoch=epoch+1,best=best,stale=stale,history=history,best_epoch=best_epoch,best_coverage=best_coverage,
                numpy_rng=rng.bit_generator.state,augmentation_rng=aux_rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                coverage=coverage.state(),exposure=exposure.rows)
            name=f'epoch_{epoch+1:03d}.pt';save_checkpoint(folder/name,saved)
            pointer=dict(name=name,sha256=sha256(folder/name),identity=key)
            write_pointer(folder/'last.json',pointer)
            if improved: write_pointer(folder/'best.json',pointer)
            save_history(folder/'training_history.csv',pd.DataFrame(history))
            save_history(folder/'regime_exposure.csv',pd.DataFrame(exposure.rows))
            print(f'[{data.panel} {variant}] epoch={epoch+1} internal={value:.9f} best_epoch={best_epoch} seconds={row["seconds"]:.1f}',flush=True)
            if stop_after_epoch==epoch+1: raise RuntimeError('Injected epoch interruption for resume validation')
        checkpoint,pointer=checkpoint_read(folder,'best.json',key);model.load_state_dict(checkpoint['state_dict'])
        selection=dict(variant=variant,best_epoch=best_epoch,best_internal_S_state=best,epochs_run=len(history),
            optimizer_updates=coverage.steps,common_initial_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
            best_checkpoint_coverage=best_coverage,final_coverage=coverage.summary(),
            selection='Same original raw-network internal regime macro S as T1; no holdout labels',config=config)
        save_json(folder/'selection.json',selection)
        names=['identity.json','best.json','last.json','selection.json','training_history.csv','regime_exposure.csv',
               pointer['name'],read(folder/'last.json')['name']]
        save_json(folder/'COMPLETE.json',dict(identity=key,artifacts=relative_snapshot(folder,names)))
        return model,dict(key=key,folder=str(folder),hit=False,best=pointer,selection=selection)


def predictions(data,model,record,dates,config,device,purpose):
    identity=dict(kind='t1_ablation_predictions_v1',panel=data.panel,model=record['key'],best=record['best']['sha256'],
        inputs=[data.records[d]['key'] for d in dates],normalization=data.provenance['normalization_key'],
        dates=dates,purpose=purpose,eval_batch=config['eval_batch'],device=str(device),implementation=code_identity())
    def build(out):
        values,_=predict(model,data,dates,config,device,False)
        keys=data.expected(dates);ii=np.concatenate([data.array(d,'ii') for d in dates])
        keys['speed_kmh']=values[:,0];keys['flow_vph']=values[:,1]*data.exp.lanes[ii]
        keys.to_parquet(out/'predictions.parquet',index=False)
        return dict(rows=len(keys),inference_augmentation=False)
    folder,manifest,hit=data.cache.get_or_build(identity,build)
    return pd.read_parquet(folder/'predictions.parquet'),dict(key=manifest['key'],folder=str(folder),hit=hit)
