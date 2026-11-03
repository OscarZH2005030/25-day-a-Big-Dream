"""Paired GPU warm-start study: state-only versus density/density-change supervision."""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,json,shutil,importlib.util,time,hashlib,gc
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'work/repo/experiments/task1/best_road_combination_20261026'
DATA=Path('C:/Users/18049/Documents/Codex/2026-10-28/https-github-com-oscarzh2005030-25-day/work/data/kaggle_public')
WORK=ROOT/'work/day1';RUNTIME=WORK/'task13-runtime';CACHE=WORK/'task13-cache'
OUT=ROOT/'outputs/day1/task13-joint';OUT.mkdir(parents=True,exist_ok=True)
ROADS=['D7_I210_W','D7_I405_S']
CONFIG=dict(seed=20261102,epochs=6,steps_per_epoch=256,learning_rate=1e-5,weight_decay=.001,
            density_weight=.05,delta_weight=.02,selection_density_weight=.02,selection_delta_weight=.01)

def save(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
def log(**value):
    save(OUT/'progress.json',dict(pid=os.getpid(),**value));print(json.dumps(value),flush=True)
def runtime():
    RUNTIME.mkdir(exist_ok=True)
    vendor=BUNDLE/'reproduction/vendor/jayden8'
    for p in vendor.iterdir():
        if p.is_file():shutil.copy2(p,RUNTIME/p.name)
    shutil.copytree(BUNDLE/'feature-source',RUNTIME/'feature-source',dirs_exist_ok=True)
    p=RUNTIME/'train_global.py';text=p.read_text(encoding='utf8');lines=[]
    for line in text.splitlines():
        if line.startswith('ROOT=Path(__file__)'):line=f"ROOT=Path(__file__).resolve().parent;WORK=Path({str(WORK/'task13-shared')!r});CACHE=Path({str(CACHE)!r})"
        elif line.startswith('DATA=Path('):line=f'DATA=Path({str(DATA)!r})'
        lines.append(line)
    text='\n'.join(lines)+'\n'
    text=text.replace('  if date in TRAIN:\n','  if False: # warm-start paired static-view study\n')
    p.write_text(text,encoding='utf8');sys.path.insert(0,str(RUNTIME))
    spec=importlib.util.spec_from_file_location('joint_vendor',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.CONFIG['augmented_view_probability']=0.
    return m

def density(pred):return torch.log1p(24*pred[...,1]/pred[...,0].clamp_min(.04))
def extra_loss(pred,batch):
    kp=density(pred);kt=density(batch['label']);mask=batch['target'].bool()
    a=F.smooth_l1_loss(kp[mask],kt[mask])
    both=mask[:,1:]&mask[:,:-1]&batch['time_valid'][:,1:,None]&batch['time_valid'][:,:-1,None]
    b=F.smooth_l1_loss((kp[:,1:]-kp[:,:-1])[both],(kt[:,1:]-kt[:,:-1])[both]) if both.any() else pred.sum()*0
    return a,b

@torch.inference_mode()
def evaluate(model,data,dates,m,export=None):
    model.eval();acc=np.zeros((3,3));den_sse=den_n=delta_sse=delta_n=0;frames=[]
    nodes=np.arange(data.n);graph=data.graph_for(nodes);starts=list(range(0,265,12))
    for date in dates:
        total=np.zeros((288,data.n,2));count=np.zeros((288,1,1))
        for offset in range(0,len(starts),4):
            ss=starts[offset:offset+4];parts=[data.block(date,s,nodes)[0] for s in ss]
            batch={k:torch.from_numpy(np.stack([v[k] for v in parts])).cuda() for k in parts[0]}
            predictions=model(batch,graph).cpu().numpy()
            for start,pred in zip(ss,predictions):total[start:start+24]+=pred;count[start:start+24]+=1
        prediction=total/count;mask=np.asarray(data.get(date,'mask'));truth=np.asarray(data.get(date,'y'));error=prediction[mask]-truth[mask]
        regime=json.loads((data.folder/date/'ready.json').read_text())['regime'];acc[regime]+=[len(error),np.square(error[:,0]).sum(),np.square(error[:,1]).sum()]
        kp=np.log1p(24*prediction[...,1]/np.maximum(prediction[...,0],.04));kt=np.log1p(24*truth[...,1]/np.maximum(truth[...,0],.04))
        den_sse+=np.square(kp[mask]-kt[mask]).sum();den_n+=mask.sum();both=mask[1:]&mask[:-1]
        delta_sse+=np.square((np.diff(kp,axis=0)-np.diff(kt,axis=0))[both]).sum();delta_n+=both.sum()
        if export:
            keys=data.keys[date].copy();ii=keys.station_id.map(data.exp.si).to_numpy(int);ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
            values=prediction[jj,ii];keys['speed_kmh']=values[:,0]*25;keys['flow_vph']=values[:,1]*600*data.exp.lanes[ii];keys.to_parquet(export/(date+'.parquet'),index=False)
    result=m.metric(acc);result.update(log_density_RMSE=float(np.sqrt(den_sse/max(den_n,1))),delta_log_density_RMSE=float(np.sqrt(delta_sse/max(delta_n,1))))
    result['selection_proxy']=result['S_state']-.02*result['log_density_RMSE']-.01*result['delta_log_density_RMSE']
    return result

def main():
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=dict(roads=ROADS,config=CONFIG,training='Published warm-start; original168trainingdates only; static masked views, no hidden labels as features.',
        selection='Same22internaldates and same state-density composite for both arms; checkpoint frozen before55day evaluation. All holdout days previously developed, not blind.',
        loss='Control originalstateRMSE. Joint adds log-perlane-density and adjacent-target log-density-change SmoothL1; only pairs where both true labels are known. This is supervised surrogate, NOT hidden-boundary LWR.',
        caveat='Two complete roads first, not full10direction score. No deployment or Kaggle upload.')
    save(OUT/'protocol.json',protocol);m=runtime();routes=json.loads((BUNDLE/'selected-road-models.json').read_text())['routes']
    for road in ROADS:
        route=routes[road];assert route['model']=='jayden_186'
        history=RUNTIME/'history'/road;history.mkdir(parents=True,exist_ok=True);shutil.copy2(BUNDLE/route['paths']['history'],history/'history.npz')
        keypath=WORK/'task13-shared/cache'/road/'keys.parquet';keypath.parent.mkdir(parents=True,exist_ok=True)
        if not keypath.exists():pd.read_csv(DATA/'task1'/road/'train/sample_submission_state.csv',dtype=str).to_parquet(keypath,index=False)
        data=m.Data(road)
        for i,date in enumerate(m.TRAIN+m.INNER):
            data.prepare(date)
            if i%10==0:log(stage='preparing',road=road,days=i+1,total=190)
        shutil.copy2(BUNDLE/route['paths']['normalization'],data.folder/'normalization.npz');data.normalization()
        for arm in ['state_control','joint_density']:
            folder=OUT/road/arm;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():continue
            torch.manual_seed(CONFIG['seed']);model=m.PriorityTransformer(route['config'],'T1_balanced').cuda()
            model.load_state_dict(torch.load(BUNDLE/route['paths']['transformer'],map_location='cuda',weights_only=False)['model'])
            optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'])
            rng=np.random.default_rng(CONFIG['seed']);aux=np.random.default_rng(1234);start_epoch=0;trace=[]
            initial=evaluate(model,data,m.INNER,m);best=initial['selection_proxy']
            if not (folder/'best.pt').exists():torch.save(model.state_dict(),folder/'best.pt')
            save(folder/'initial.json',initial)
            if (folder/'last.pt').exists():
                ck=torch.load(folder/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];aux.bit_generator.state=ck['aux'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']]);start_epoch=ck['epoch'];best=ck['best'];trace=ck['trace']
            for epoch in range(start_epoch,CONFIG['epochs']):
                model.train();losses=[]
                for step in range(CONFIG['steps_per_epoch']):
                    batch,graph,_=data.batch(rng,aux);optimizer.zero_grad(set_to_none=True);prediction=model(batch,graph);loss=m.objective(prediction,batch)
                    if arm=='joint_density':
                        a,b=extra_loss(prediction,batch);loss=loss+.05*a+.02*b
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
                score=evaluate(model,data,m.INNER,m)
                if score['selection_proxy']>best+1e-8:best=score['selection_proxy'];torch.save(model.state_dict(),folder/'best.pt')
                trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));save(folder/'trace.json',trace)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,aux=aux.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),epoch=epoch+1,best=best,trace=trace),folder/'last.pt')
                log(stage='epoch',road=road,arm=arm,epoch=epoch+1,**score)
            save(folder/'frozen-before-holdout.json',dict(checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),selection_dates=m.INNER))
            data.allow_holdout=True
            for date in m.HOLD:data.prepare(date)
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False));pred=folder/'holdout-predictions';pred.mkdir(exist_ok=True)
            result=evaluate(model,data,m.HOLD,m,pred);save(folder/'complete.json',dict(road=road,arm=arm,**result));log(stage='road_arm_complete',road=road,arm=arm,**result)
            del optimizer,model;gc.collect();torch.cuda.empty_cache()
        del data;gc.collect()
    save(OUT/'complete.json',dict(complete=True,official_score=None))

if __name__=='__main__':main()
