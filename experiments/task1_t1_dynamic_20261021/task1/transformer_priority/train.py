from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from .common import METHODS, SCALE, cache_key, metric_rows, read, save_json, sha256, snapshot, upstream
from .data import graph_tensors
from .model import model_for, objective
from ..wave_neural.cache import save_checkpoint, save_history


class Coverage:
    def __init__(self,data,state=None):
        self.data=data;self.visited={};self.visits=0;self.empty_steps=0;self.steps=0
        self.sampling_sha256='0'*64
        self.group_visits={k:0 for k in ['all','time_edge','road_edge','joint_edge','interior']}
        self.road=np.zeros(data.n,dtype=bool);width=min(8,max(1,data.n//2))
        self.road[data.graph['order'][:width]]=True;self.road[data.graph['order'][-width:]]=True
        self.clock=np.zeros(288,dtype=bool);self.clock[:12]=True;self.clock[-12:]=True
        if state is not None:
            for date,packed in state['visited'].items():
                self.visited[date]=np.unpackbits(packed,count=288*data.n).reshape(288,data.n).astype(bool)
            for key in ['visits','empty_steps','steps','sampling_sha256','group_visits']:
                setattr(self,key,state[key])

    def update(self,batch,spec):
        nodes=np.asarray(spec['nodes'])
        total=0
        for i,(date,start) in enumerate(spec['samples']):
            mask=batch['target'][i].numpy();tt,nn=np.nonzero(mask)
            ticks=tt+start;selected=nodes[nn]
            if len(ticks) and ((ticks<0).any() or (ticks>=288).any()):
                raise ValueError('Padded position received supervision')
            if date not in self.visited:
                self.visited[date]=np.zeros((288,self.data.n),dtype=bool)
            self.visited[date][ticks,selected]=True
            time_edge=self.clock[ticks];road_edge=self.road[selected]
            counts=dict(all=len(tt),time_edge=int(time_edge.sum()),road_edge=int(road_edge.sum()),
                        joint_edge=int((time_edge&road_edge).sum()),interior=int((~time_edge&~road_edge).sum()))
            for key,value in counts.items():
                self.group_visits[key]+=value
            total+=len(tt)
        self.visits+=total;self.empty_steps+=int(total==0);self.steps+=1
        self.sampling_sha256=hashlib.sha256(bytes.fromhex(self.sampling_sha256)+json.dumps(spec,sort_keys=True).encode()).hexdigest()

    def state(self):
        return dict(visited={d:np.packbits(v) for d,v in self.visited.items()},visits=self.visits,empty_steps=self.empty_steps,
                    steps=self.steps,sampling_sha256=self.sampling_sha256,group_visits=self.group_visits)

    def summary(self):
        available={k:0 for k in self.group_visits};unique=available.copy()
        for date in self.data.roles['training']:
            target=self.data.array(date,'target');visited=self.visited.get(date,np.zeros_like(target))
            if (visited&~target).any():
                raise ValueError('Coverage contains non-template targets')
            time_edge=self.clock[:,None];road_edge=self.road[None,:]
            groups=dict(all=np.ones_like(target),time_edge=np.broadcast_to(time_edge,target.shape),
                        road_edge=np.broadcast_to(road_edge,target.shape),joint_edge=time_edge&road_edge,interior=~time_edge&~road_edge)
            for name,mask in groups.items():
                available[name]+=int((target&mask).sum());unique[name]+=int((visited&mask).sum())
        return dict(available_targets=available['all'],unique_targets_seen=unique['all'],target_visits=self.visits,
                    steps=self.steps,empty_steps=self.empty_steps,sampling_sha256=self.sampling_sha256,
                    groups={k:dict(available=available[k],unique=unique[k],visits=self.group_visits[k],
                                   unique_fraction=unique[k]/max(available[k],1),mean_visits=self.group_visits[k]/max(available[k],1)) for k in available})


@torch.no_grad()
def predict(model,data,dates,config,device,diagnostics=False):
    model.eval();window=config['window'];nodes=np.arange(data.n)
    graph={k:v.to(device) for k,v in graph_tensors(data.graph,nodes).items()}
    starts=list(range(0,289-window,window//2))
    if starts[-1]!=288-window:
        starts.append(288-window)
    predictions=[];diagnostic=[]
    offsets=list(range(0,len(starts),config.get('eval_batch',2)))
    diagnostic_offsets={offsets[0],offsets[len(offsets)//2],offsets[-1]}
    for date in dates:
        total=np.zeros((288,data.n,2),dtype=np.float64);count=np.zeros((288,1,1),dtype=np.float64)
        for offset in range(0,len(starts),config.get('eval_batch',2)):
            ticks=starts[offset:offset+config.get('eval_batch',2)]
            parts=[data.block(date,t,nodes,window,False) for t in ticks]
            batch={k:torch.stack([p[k] for p in parts]).to(device) for k in parts[0]}
            collect=diagnostics and offset in diagnostic_offsets
            values=model(batch,graph,diagnostics=collect).cpu().numpy()
            for t,value in zip(ticks,values):
                total[t:t+window]+=value;count[t:t+window]+=1
            if collect:
                diagnostic.append(dict(date=date,window_starts=ticks,scope='Fixed first/middle/last window batches per date; descriptive sample',blocks=[b.diagnostic for b in model.blocks]))
        if not (count>0).all():
            raise ValueError('Inference windows missed a real timestamp')
        grid=total/count
        predictions.append(grid[data.array(date,'jj'),data.array(date,'ii')]*SCALE)
    return np.concatenate(predictions),diagnostic


def internal_score(model,data,config,device):
    dates=data.roles['internal'];prediction,_=predict(model,data,dates,config,device)
    truth=data.truth(dates,'internal');keys=data.expected(dates)
    rows=metric_rows(truth,prediction,keys.mask_regime.to_numpy(),'network',data.panel,data.family)
    return float(np.mean([r['S_state'] for r in rows]))


def checkpoint_read(folder,pointer,identity):
    item=read(folder/pointer)
    path=(folder/item['name']).resolve()
    if not path.is_relative_to(folder.resolve()) or sha256(path)!=item['sha256'] or item['identity']!=identity:
        raise ValueError('Checkpoint hash/identity mismatch')
    return torch.load(path,map_location='cpu',weights_only=False),item


def fit(data,config,method,device,cache_root,source_identity,status):
    identity=dict(kind='tt_priority_model',panel=data.panel,method=method,config=config,
                  feature_days=[data.records[d]['key'] for d in data.roles['training']+data.roles['internal']],
                  normalization=data.provenance['normalization_key'],roles={k:data.roles[k] for k in ['training','internal']},source=source_identity)
    key=cache_key(identity);folder=cache_root/'models'/key;folder.mkdir(parents=True,exist_ok=True)
    model,initial_hash=model_for(config,method)
    with upstream().lock(folder/'worker.lock'):
        if (folder/'identity.json').exists():
            if read(folder/'identity.json')!=identity:
                raise ValueError('Existing training identity differs')
        else:
            save_json(folder/'identity.json',identity)
        if (folder/'COMPLETE.json').exists():
            complete=read(folder/'COMPLETE.json')
            for item in complete['artifacts']:
                if sha256(Path(item['path']))!=item['sha256']:
                    raise ValueError('Completed model artifact changed')
            checkpoint,best=checkpoint_read(folder,'best.json',key)
            model.load_state_dict(checkpoint['state_dict'])
            print('[model HIT]',data.panel,method,key[:12],flush=True)
            return model.to(device),dict(key=key,folder=str(folder),hit=True,selection=read(folder/'selection.json'),best=best)
        model=model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs']*config['steps_per_epoch'],eta_min=config['learning_rate']*.1)
        rng=np.random.default_rng(config['seed']);coverage=Coverage(data)
        best=-float('inf');stale=0;start_epoch=0;history=[];best_epoch=None;best_coverage=None
        if (folder/'last.json').exists():
            saved,_=checkpoint_read(folder,'last.json',key)
            model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
            if device.type=='cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            coverage=Coverage(data,saved['coverage']);start_epoch=saved['epoch'];best=saved['best'];stale=saved['stale']
            history=saved['history'];best_epoch=saved['best_epoch'];best_coverage=saved['best_coverage']
            print('[resume]',data.panel,method,'epoch',start_epoch,flush=True)
        else:
            torch.manual_seed(config['seed']+123)
            if device.type=='cuda':
                torch.cuda.manual_seed_all(config['seed']+123)
        for epoch in range(start_epoch,config['epochs']):
            if stale>=config['patience']:
                break
            started=time.perf_counter();model.train();losses=[]
            for step in range(config['steps_per_epoch']):
                batch,graph,spec=data.batch(rng,config,method!=METHODS[0]);coverage.update(batch,spec)
                batch={k:v.to(device) for k,v in batch.items()};graph={k:v.to(device) for k,v in graph.items()}
                optimizer.zero_grad(set_to_none=True)
                loss=objective(model(batch,graph),batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'],error_if_nonfinite=True)
                optimizer.step();scheduler.step();losses.append(float(loss.detach()))
                if step==0 or (step+1)%128==0:
                    status(panel=data.panel,method=method,stage='training',epoch=epoch+1,step=step+1)
                    print(f'[{data.panel} {method}] epoch={epoch+1} step={step+1}/{config["steps_per_epoch"]} loss={np.mean(losses):.6f}',flush=True)
            status(panel=data.panel,method=method,stage='internal_validation',epoch=epoch+1,step=config['steps_per_epoch'])
            value=internal_score(model,data,config,device);improved=value>best;stale=0 if improved else stale+1
            if improved:
                best=value;best_epoch=epoch+1;best_coverage=coverage.summary()
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),internal_S_state=value,best_S_state=best,
                     seconds=time.perf_counter()-started,sampling_sha256=coverage.sampling_sha256,unique_targets_seen=coverage.summary()['unique_targets_seen'])
            history.append(row)
            saved=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,
                       best=best,stale=stale,history=history,best_epoch=best_epoch,best_coverage=best_coverage,
                       numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,coverage=coverage.state())
            name=f'epoch_{epoch+1:03d}.pt';save_checkpoint(folder/name,saved)
            pointer=dict(name=name,sha256=sha256(folder/name),identity=key)
            save_json(folder/'last.json',pointer)
            if improved:
                save_json(folder/'best.json',pointer)
            save_history(folder/'training_history.csv',pd.DataFrame(history))
            print(f'[{data.panel} {method}] epoch={epoch+1} internal={value:.9f} best_epoch={best_epoch} seconds={row["seconds"]:.1f}',flush=True)
        checkpoint,pointer=checkpoint_read(folder,'best.json',key);model.load_state_dict(checkpoint['state_dict'])
        selection=dict(method=method,best_epoch=best_epoch,best_internal_S_state=best,epochs_run=len(history),steps=coverage.steps,
                       common_initial_sha256=initial_hash,parameters=sum(p.numel() for p in model.parameters()),
                       best_checkpoint_coverage=best_coverage,final_coverage=coverage.summary(),
                       selection='Original raw-network regime macro S_state on22 internal dates; no holdout selection')
        save_json(folder/'selection.json',selection)
        artifacts=snapshot([folder/n for n in ['identity.json','last.json','best.json','training_history.csv','selection.json',pointer['name']]])
        save_json(folder/'COMPLETE.json',dict(identity=key,artifacts=artifacts,all_training_finished=True))
        return model,dict(key=key,folder=str(folder),hit=False,selection=selection,best=pointer)


def cached_predictions(data,model,record,dates,config,device,purpose):
    identity=dict(kind='tt_priority_predictions',panel=data.panel,model=record['key'],best=record['best']['sha256'],
                  inputs=[data.records[d]['key'] for d in dates],normalization=data.provenance['normalization_key'],purpose=purpose,eval_batch=config['eval_batch'])
    def build(out):
        values,diagnostic=predict(model,data,dates,config,device,True)
        keys=data.expected(dates);ii=np.concatenate([data.array(d,'ii') for d in dates])
        keys['speed_kmh']=values[:,0];keys['flow_vph']=values[:,1]*data.exp.lanes[ii]
        keys.to_parquet(out/'predictions.parquet',index=False)
        save_json(out/'attention_diagnostics.json',diagnostic)
        return dict(rows=len(keys),dates=dates)
    folder,manifest,hit=data.cache.get_or_build(identity,build)
    return pd.read_parquet(folder/'predictions.parquet'),dict(key=manifest['key'],folder=str(folder),hit=hit)
