"""CPU inference only: fixed >60-minute endpoint-span routing on internal dates."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(60,repeat=True)
print('Loading PyTorch (CPU inference only)...',flush=True)
import torch
print('Loading LightGBM...',flush=True)
import lightgbm as lgb
import numpy as np
import pandas as pd
print('Libraries loaded.',flush=True)
faulthandler.cancel_dump_traceback_later()

SOURCE = Path(__file__).resolve().parents[1]/'full_dataset_training_v2'
TF_RUN = SOURCE/'runs/all_corridors_time80_20'
LGB_RUN = Path(__file__).resolve().parents[1]/'lightgbm_time80_20/runs/all_corridors_time80_20'
KEYS = ['panel','timestamp','station_id','link_id','mask_regime']
SCALE = np.array([25.,600.])
sys.path.insert(0,str(SOURCE))
from full_train_worker import StateTransformer, MemoryDayStore
from task1_baseline import Experiment, score
from training_io import RunLock


def write_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def route(prev_gap,next_gap):
    bounded=np.isfinite(prev_gap)&np.isfinite(next_gap)
    span=np.where(bounded,prev_gap+next_gap,np.nan)
    # Strictly greater than 60. One missing endpoint always falls back to LGB.
    long=bounded&(span>60)
    group=np.where(~bounded,'one_or_both_sides_missing',np.where(long,'span_over60min','span_at_most60min'))
    return long,span,group


def make_graph(exp):
    n=exp.n;topology=pd.read_csv(exp.panel/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
    adj=np.eye(n,dtype=bool)
    for groups in exp.neighbors.values():
        for i,ids in enumerate(groups):adj[i,ids]=True
    adj=adj|adj.T
    store=MemoryDayStore.__new__(MemoryDayStore)
    store.graph=adj|((adj.astype(int)@adj.astype(int))>0)
    order=np.array([topology.loc[l,'order_index'] for l in exp.links])
    store.direction=np.sign(order[None,:]-order[:,None])
    store.up=np.array([ids[0] if len(ids)==1 else -1 for ids in exp.neighbors['up']])
    store.static=exp.static[['lanes','length_km','free_speed_kmh','capacity_vph','critical_density','k_jam']].to_numpy(dtype='float32')
    return store.graph_tensors(np.arange(n))


@torch.inference_mode()
def predict_day(model,raw,base,mean,scale,graph,window,batch_windows):
    finite=np.isfinite(raw)
    x=np.concatenate([np.where(finite,(raw-mean)/scale,0),finite],axis=-1).astype('float32')
    starts=list(range(0,289-window,max(1,window//2)))
    if starts[-1]!=288-window:starts.append(288-window)
    total=np.zeros(base.shape,dtype='float64');weight=np.zeros((288,1,1),dtype='float64')
    for k in range(0,len(starts),batch_windows):
        indices=starts[k:k+batch_windows]
        batch={'features':torch.from_numpy(np.stack([x[t:t+window] for t in indices])),
               'base':torch.from_numpy(np.stack([base[t:t+window] for t in indices]))}
        values=model(batch,graph).numpy()
        for j,t in enumerate(indices):total[t:t+window]+=values[j];weight[t:t+window]+=1
    return total/weight


def evaluate_panel(panel,args,plan):
    started=time.time();folder=args.output/panel;folder.mkdir(exist_ok=True)
    reference=TF_RUN/panel;cache=reference/'cache'
    checkpoint_path=reference/'best.pt'
    ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    config=SimpleNamespace(**ck['args']);config.output=str(folder)
    exp=Experiment(config)
    assert exp.stations==list(ck['stations']) and exp.links==list(ck['links'])
    meta=json.loads((cache/'ready.json').read_text());names=meta['names'];mean=np.asarray(meta['mean']);scale=np.asarray(meta['scale'])
    assert ck['input_names']==names+[n+'__finite' for n in names]
    np.testing.assert_allclose(mean,ck['feature_mean'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(scale,ck['feature_scale'],rtol=0,atol=1e-10)
    model=StateTransformer(102,config.width,config.heads,config.layers,config.window,config.dropout)
    model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False)
    graph=make_graph(exp)
    trees=[lgb.Booster(model_file=str(LGB_RUN/panel/(n+'.txt'))) for n in ['speed_kmh','flow_per_lane']]
    frames=[];truths=[];pred_tf=[];pred_lgb=[];masks=[];groups=[];spans=[];raw_sse=np.zeros(2);n_raw=0
    for index,date in enumerate(plan['internal_validation_dates']):
        print(f'{panel}: INTERNAL {date} ({index+1}/22), CPU inference only',flush=True)
        f=cache/date;raw=np.load(f/'features.npy',mmap_mode='r');base=np.load(f/'base.npy',mmap_mode='r')
        ii=np.load(f/'ii.npy');jj=np.load(f/'jj.npy');keys=pd.read_parquet(f/'keys.parquet')
        assert keys.timestamp.astype(str).str[:10].eq(date).all()
        assert keys.panel.eq(panel).all() and len(keys)==len(ii)
        target=np.load(f/'target.npy',mmap_mode='r')
        assert int(target.sum())==len(ii) and target[jj,ii].all()
        neural=predict_day(model,raw,base,mean,scale,graph,config.window,args.batch_windows)[jj,ii]*SCALE
        x=raw[jj,ii];b=base[jj,ii].astype('float64')*SCALE
        residual=np.column_stack([t.predict(x,num_threads=args.threads) for t in trees])
        truth=np.load(f/'label.npy',mmap_mode='r')[jj,ii].astype('float64')*SCALE
        raw_sse+=((b+residual-truth)**2).sum(axis=0);n_raw+=len(ii)
        tree=np.maximum(b+residual,0)
        channel_masks=[];channel_groups=[];channel_spans=[]
        for col in ['speed_kmh','flow_vph']:
            long,span,group=route(x[:,names.index(col+'_prev_gap_min')],x[:,names.index(col+'_next_gap_min')])
            channel_masks.append(long);channel_groups.append(group);channel_spans.append(span)
        frames.append(keys);truths.append(truth);pred_tf.append(neural);pred_lgb.append(tree)
        masks.append(np.column_stack(channel_masks));groups.append(np.column_stack(channel_groups));spans.append(np.column_stack(channel_spans))
    keys=pd.concat(frames,ignore_index=True);y=np.concatenate(truths);tf=np.concatenate(pred_tf);tree=np.concatenate(pred_lgb)
    use_tf=np.concatenate(masks);group=np.concatenate(groups);span=np.concatenate(spans);regimes=keys.mask_regime.to_numpy()
    assert not keys.duplicated(KEYS).any()
    hybrid=np.where(use_tf,tf,tree)
    speed_only=np.column_stack([hybrid[:,0],tree[:,1]])
    flow_only=np.column_stack([tree[:,0],hybrid[:,1]])
    methods={'LightGBM':tree,'Transformer':tf,'hybrid_both':hybrid,'hybrid_speed_only':speed_only,'hybrid_flow_only':flow_only}
    metrics=[]
    for name,pred in methods.items():
        assert np.isfinite(pred).all() and (pred>=0).all()
        metrics+=score(y,pred,regimes,name)
    frame=pd.DataFrame(metrics)
    internal_tf=float(frame[(frame.method=='Transformer')&(frame.regime=='macro_mean')].S_state.iloc[0])
    tolerance=2e-5
    if abs(internal_tf-float(ck['best']))>tolerance:
        raise ValueError(f'Internal Transformer score differs from checkpoint best: {internal_tf} vs {ck["best"]}; stopping')
    choices=json.loads((LGB_RUN/panel/'experiment.json').read_text())['choices']
    tree_rmse=np.sqrt(raw_sse/n_raw);expected=np.array([c['internal_residual_rmse'] for c in choices])
    np.testing.assert_allclose(tree_rmse,expected,rtol=1e-5,atol=1e-4)
    rows=[]
    for c,channel in enumerate(['speed','flow_per_lane']):
        for category in ['span_at_most60min','span_over60min','one_or_both_sides_missing','all']:
            for regime in ['R1','R2','R3','pooled']:
                keep=(np.ones(len(y),bool) if category=='all' else group[:,c]==category)
                if regime!='pooled':keep&=regimes==regime
                if not keep.any():continue
                for name,pred in [('LightGBM',tree),('Transformer',tf),('hybrid',hybrid)]:
                    rmse=float(np.sqrt(np.mean((pred[keep,c]-y[keep,c])**2)))
                    rows.append(dict(channel=channel,gap_group=category,regime=regime,method=name,
                                     n=int(keep.sum()),rmse=rmse,S_channel=max(0,1-rmse/SCALE[c])))
    frame.to_csv(folder/'metrics.csv',index=False);pd.DataFrame(rows).to_csv(folder/'channel_metrics.csv',index=False)
    result=keys.copy()
    for c,channel in enumerate(['speed','flow_per_lane']):
        result[channel+'_truth']=y[:,c];result[channel+'_lgb']=tree[:,c];result[channel+'_tf']=tf[:,c]
        result[channel+'_hybrid']=hybrid[:,c];result[channel+'_use_tf']=use_tf[:,c];result[channel+'_endpoint_span_min']=span[:,c]
    result.to_parquet(folder/'internal_predictions.parquet',index=False)
    counts={channel:{category:int((group[:,c]==category).sum()) for category in ['span_at_most60min','span_over60min','one_or_both_sides_missing']}
            for c,channel in enumerate(['speed','flow_per_lane'])}
    write_json(folder/'COMPLETE.json',dict(panel=panel,n=len(y),counts=counts,elapsed_seconds=time.time()-started,
             transformer_best_internal_score=float(ck['best']),recomputed_internal_score=internal_tf,
             transformer_score_difference=internal_tf-float(ck['best']),lgb_raw_rmse=tree_rmse.tolist(),
             checkpoint_sha256=sha(checkpoint_path),lgb_model_sha256=[sha(LGB_RUN/panel/(n+'.txt')) for n in ['speed_kmh','flow_per_lane']]))
    print('FINISHED '+panel+'\n'+frame[frame.regime=='macro_mean'][['method','S_state']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch-windows',type=int,default=4)
    args=parser.parse_args()
    if min(args.threads,args.batch_windows)<1:parser.error('threads and batch-windows must be positive')
    # Boundary contract: 60 stays with LGB; 65 uses TF; a missing side stays LGB.
    long,span,group=route(np.array([5.,30.,30.,np.nan]),np.array([55.,30.,35.,90.]))
    assert long.tolist()==[False,False,True,False]
    torch.set_num_threads(args.threads);torch.set_grad_enabled(False)
    plan=json.loads((TF_RUN/'plan.json').read_text())
    config=dict(rule='both finite endpoint gaps and prev_gap_min + next_gap_min > 60; else LightGBM',
                dates=plan['internal_validation_dates'],panels=plan['panels'],device='cpu',training=False,
                threshold_tuned=False,script_sha256=sha(Path(__file__)),threads=args.threads,batch_windows=args.batch_windows)
    args.output.mkdir(parents=True,exist_ok=True)
    with RunLock(args.output/'evaluation.lock'):
        manifest=args.output/'plan.json'
        if manifest.exists() and json.loads(manifest.read_text())!=config:raise ValueError('Existing output configuration differs')
        write_json(manifest,config);complete=[]
        try:
            for panel in plan['panels']:
                if (args.output/panel/'COMPLETE.json').exists():complete.append(panel);continue
                write_json(args.output/'status.json',dict(state='running',current=panel,completed=complete))
                evaluate_panel(panel,args,plan);complete.append(panel)
            metrics=[];channels=[];counts=[]
            for panel in plan['panels']:
                for filename,target in [('metrics.csv',metrics),('channel_metrics.csv',channels)]:
                    frame=pd.read_csv(args.output/panel/filename);frame.insert(0,'panel',panel);target.append(frame)
                record=json.loads((args.output/panel/'COMPLETE.json').read_text())
                for channel,values in record['counts'].items():
                    for category,n in values.items():counts.append(dict(panel=panel,channel=channel,gap_group=category,n=n))
            all_metrics=pd.concat(metrics);all_channels=pd.concat(channels)
            all_metrics.to_csv(args.output/'all_metrics.csv',index=False);all_channels.to_csv(args.output/'all_channel_metrics.csv',index=False)
            pd.DataFrame(counts).to_csv(args.output/'routing_counts.csv',index=False)
            comparison=all_metrics[all_metrics.regime=='macro_mean'].pivot(index='panel',columns='method',values='S_state')
            comparison['hybrid_minus_lgb']=comparison.hybrid_both-comparison.LightGBM
            comparison['hybrid_minus_tf']=comparison.hybrid_both-comparison.Transformer
            comparison.to_csv(args.output/'comparison.csv')
            write_json(args.output/'summary.json',dict(mean_corridor_S_state=comparison.mean().to_dict(),
                total_internal_targets=int(pd.DataFrame(counts).query("channel == 'speed'").n.sum()),
                note='Internal development validation; checkpoints already selected here. Fixed routing test, no new training or holdout evaluation.'))
            write_json(args.output/'status.json',dict(state='complete',current=None,completed=complete))
            print(comparison.to_string(),flush=True)
        except BaseException as e:
            write_json(args.output/'status.json',dict(state='failed',current=panel,completed=complete,error=str(e)));raise


if __name__=='__main__':main()
