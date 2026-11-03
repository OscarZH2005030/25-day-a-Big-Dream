"""Paired warm starts with physical-unit relative accumulation-change L1 loss."""
import os,sys,json,hashlib,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
import task13_joint as base
OUT=ROOT/'outputs/day1/task13-nratio';OUT.mkdir(exist_ok=True)
CFG=dict(seed=20261102,epochs=6,steps=256,lr=1e-5,state_weight=.35,N_weight=.1)
ARMS=['state_control','joint_Nratio']
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')
def log(**v):save(OUT/'progress.json',dict(pid=os.getpid(),**v));print(json.dumps(v),flush=True)
def accumulation(p,scale):return 24*p[...,1]/p[...,0].clamp_min(.04)*scale[:,None,:]
def ratio(pred,b):
    p=accumulation(pred,b['physical_scale']);y=accumulation(b['label'],b['physical_scale'])
    mask=b['target'].bool();both=mask[:,1:]&mask[:,:-1]&b['time_valid'][:,1:,None]&b['time_valid'][:,:-1,None]
    if not both.any():return pred.sum()*0
    dp=p[:,1:]-p[:,:-1];dy=y[:,1:]-y[:,:-1]
    return (dp[both]-dy[both]).abs().sum()/dy[both].abs().sum().clamp_min(1e-3)

@torch.inference_mode()
def evaluate(model,data,dates,m,folder=None):
    model.eval();acc=np.zeros((3,3));num=np.zeros(3);den=np.zeros(3);nodes=np.arange(data.n);graph=data.graph_for(nodes);starts=list(range(0,265,12))
    physical=data.physical
    for date in dates:
        total=np.zeros((288,data.n,2));count=np.zeros((288,1,1))
        for offset in range(0,len(starts),4):
            ss=starts[offset:offset+4];parts=[data.block(date,s,nodes)[0] for s in ss];b={k:torch.from_numpy(np.stack([x[k] for x in parts])).cuda() for k in parts[0]};p=model(b,graph).cpu().numpy()
            for start,value in zip(ss,p):total[start:start+24]+=value;count[start:start+24]+=1
        pred=total/count;mask=np.asarray(data.get(date,'mask'));y=np.asarray(data.get(date,'y'));e=pred[mask]-y[mask];reg=json.loads((data.folder/date/'ready.json').read_text())['regime'];acc[reg]+=[len(e),np.square(e[:,0]).sum(),np.square(e[:,1]).sum()]
        npred=24*pred[...,1]/np.maximum(pred[...,0],.04)*physical[None];ntrue=24*y[...,1]/np.maximum(y[...,0],.04)*physical[None];both=mask[1:]&mask[:-1]
        dp=np.diff(npred,axis=0)[both];dy=np.diff(ntrue,axis=0)[both];num[reg]+=np.abs(dp-dy).sum();den[reg]+=np.abs(dy).sum()
        if folder:
            keys=data.keys[date].copy();ii=keys.station_id.map(data.exp.si).to_numpy(int);ts=pd.to_datetime(keys.timestamp,utc=True);jj=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int);values=pred[jj,ii]
            keys['speed_kmh']=values[:,0]*25;keys['flow_vph']=values[:,1]*600*data.exp.lanes[ii];keys.to_parquet(folder/(data.road+'_'+date+'.parquet'),index=False)
    result=m.metric(acc);score=float(np.maximum(0,1-num/np.maximum(den,1e-9)).mean());result.update(N_change_score_proxy=score,N_error_ratios=(num/np.maximum(den,1e-9)).tolist(),selection_proxy=.35*result['S_state']+.1*score)
    return result

def main():
    assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=dict(config=CFG,roads=base.ROADS,training='Same168trainingdays,22internalcheckpointselection,55previouslyviewedholdout. No newdate/road/caseexceptions.',
        N='Actual per-link total accumulation:24*q_norm/v_norm * lanes*length; L1 delta error normalized by true delta magnitude. Known adjacent target pairs only, never unknown labels.',
        selection='Same .35state+.10clippedNchangeproxy for both arms. This is NOT hidden-boundary official LWR and excludes non-target adjacent transitions. Freeze checkpoint before holdout.',no_upload=True)
    save(OUT/'protocol.json',protocol);m=base.runtime();routes=json.loads((base.BUNDLE/'selected-road-models.json').read_text())['routes']
    for road in base.ROADS:
        data=m.Data(road);data.normalization();links=pd.read_csv(base.DATA/'corridors'/road/'network/links.csv',dtype={'link_id':str}).drop_duplicates('link_id').set_index('link_id')
        data.physical=(links.reindex(data.exp.links).length_km.to_numpy()*data.exp.lanes).astype('float32');assert np.isfinite(data.physical).all()
        original=data.block
        def block(date,start,nodes,aux=None):
            b,info=original(date,start,nodes,aux);b['physical_scale']=data.physical[nodes];return b,info
        data.block=block;route=routes[road]
        for arm in ARMS:
            folder=OUT/road/arm;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():continue
            torch.manual_seed(CFG['seed']);model=m.PriorityTransformer(route['config'],'T1_balanced').cuda();model.load_state_dict(torch.load(base.BUNDLE/route['paths']['transformer'],map_location='cuda',weights_only=False)['model'])
            opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=.001);rng=np.random.default_rng(CFG['seed']);aux=np.random.default_rng(1234);trace=[];first=0
            initial=evaluate(model,data,m.INNER,m);best=initial['selection_proxy'];save(folder/'initial.json',initial)
            if not (folder/'best.pt').exists():torch.save(model.state_dict(),folder/'best.pt')
            if (folder/'last.pt').exists():
                ck=torch.load(folder/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.bit_generator.state=ck['rng'];aux.bit_generator.state=ck['aux'];first=ck['epoch'];best=ck['best'];trace=ck['trace'];torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
            for epoch in range(first,CFG['epochs']):
                model.train();losses=[]
                for _ in range(CFG['steps']):
                    b,g,_=data.batch(rng,aux);opt.zero_grad(set_to_none=True);p=model(b,g);loss=.35*m.objective(p,b)
                    if arm=='joint_Nratio':loss=loss+.1*ratio(p,b)
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
                score=evaluate(model,data,m.INNER,m)
                if score['selection_proxy']>best+1e-8:best=score['selection_proxy'];torch.save(model.state_dict(),folder/'best.pt')
                trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**score));save(folder/'trace.json',trace);torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,aux=aux.bit_generator.state,epoch=epoch+1,best=best,trace=trace,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),folder/'last.pt');log(road=road,arm=arm,**trace[-1])
            save(folder/'frozen-before-holdout.json',dict(sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest()))
            data.allow_holdout=True
            for date in m.HOLD:data.prepare(date)
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False));pred=folder/'holdout-predictions';pred.mkdir(exist_ok=True)
            score=evaluate(model,data,m.HOLD,m,pred);save(folder/'complete.json',dict(road=road,arm=arm,**score));del model,opt;gc.collect();torch.cuda.empty_cache()
    save(OUT/'complete.json',dict(complete=True,official_score=None))

if __name__=='__main__':main()
