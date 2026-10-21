"""Portable learned-profile export and candidate submission inference.

Original competition files are required; no held-out labels are read here.
The previous submission remains unchanged. New outputs go to a separate folder.
"""
import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from prepare_trafficflow_data import grid
from prepare_features import FEATURES,context,neighbor,ramps
from hybrid_model import make_model
from quality_features import QUALITY_FEATURES,quality_features


def export_assets(cache,destination):
    metadata=json.loads((cache/'metadata.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    manifest=dict(features=metadata['features'],panels=[])
    offset=0
    for p in metadata['panels']:
        folder=destination/p['panel'];folder.mkdir(exist_ok=True)
        for name in ['historical-profile.npz','graph.npz']:
            shutil.copyfile(cache/p['panel']/name,folder/name)
        manifest['panels'].append(dict(panel=p['panel'],family=p['family'],links=p['links'],nodes=p['nodes'],offset=offset))
        offset+=p['nodes']
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def features_for_day(frame,day,pm,static,profile,g,ramp_path,ramp_mapping):
    links=pm['links'];n=len(links)
    speed,flow,quality=grid(frame,links,['speed_kmh','flow_vph','pct_observed'])
    lanes=static.lanes.to_numpy(np.float32);free=static.free_speed_kmh.to_numpy(np.float32)
    capacity=static.capacity_vph.to_numpy(np.float32)/lanes
    # Match the original raw-grid cache quantization exactly.
    raw=np.stack([np.nan_to_num(speed)/100,np.nan_to_num(flow/lanes)/2000],-1).astype(np.float16).astype(np.float32)
    quality=(quality/100).astype(np.float16).astype(np.float32)
    valid=np.isfinite(speed)&np.isfinite(flow)&(quality>=.75)
    weekday=dt.date.fromisoformat(day).weekday()
    count=profile['counts'][weekday]
    history=np.where(count>0,profile['sums'][weekday]/np.maximum(count,1),profile['fallback']).astype(np.float32)
    history_scaled=history/np.array([4.,2000/600],np.float32)
    values=np.where(valid[...,None],raw,0)
    initial,prev,nxt,gb,ga,span,hb,ha=context(values,valid,history_scaled)
    up,um=neighbor(values,valid,g['up_index'],g['up_weight'])
    down,dm=neighbor(values,valid,g['down_index'],g['down_weight'])
    up2,_=neighbor(up,um,g['up_index'],g['up_weight'])
    down2,_=neighbor(down,dm,g['down_index'],g['down_weight'])
    ramp=ramps(ramp_path,ramp_mapping,links,lanes)
    tod=np.arange(288,dtype=np.float32)[:,None]*2*np.pi/288
    fraction=valid.mean(1,keepdims=True)
    global_values=values.sum(1)/np.maximum(valid.sum(1,keepdims=True),1)
    position=np.cumsum(static.length_km.to_numpy());position/=position[-1]
    channels=[initial[...,0],initial[...,1],history_scaled[...,0],history_scaled[...,1],
       values[...,0],values[...,1],valid,quality,prev[...,0],nxt[...,0],prev[...,1],nxt[...,1],
       gb,ga,span,hb,ha,up[...,0],up[...,1],um,down[...,0],down[...,1],dm,
       up2[...,0],up2[...,1],down2[...,0],down2[...,1],
       ramp[...,0],ramp[...,1],ramp[...,2],ramp[...,3],
       free/120,capacity/2400,lanes/8,static.length_km.to_numpy()/5,
       np.sin(tod),np.cos(tod),np.sin(weekday*2*np.pi/7),np.cos(weekday*2*np.pi/7),
       position,global_values[:,0,None],global_values[:,1,None],fraction,
       raw[...,0],raw[...,1],initial[...,0]*100/free,initial[...,1]*2000/capacity,
       np.minimum(count[...,0],32)/32]
    assert len(channels)==len(FEATURES)
    output=np.stack([np.broadcast_to(v,(288,n)) for v in channels]).astype(np.float16)
    assert np.isfinite(output).all()
    return output


def predict(args):
    torch.set_num_threads(4)
    route_manifest=json.loads(args.route_manifest.read_text())
    routes={v['panel']:v for v in route_manifest['routes']}
    checkpoint=torch.load(args.checkpoint,weights_only=True,map_location='cuda')
    config=checkpoint['config']
    feature_names=config['features']
    assert feature_names in [FEATURES,FEATURES+QUALITY_FEATURES]
    if args.assets is None:args.assets=ROOT/('inference-assets-64' if len(feature_names)==64 else 'inference-assets')
    model=make_model(config,len(feature_names)).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    calibration=json.loads(args.calibration.read_text())
    alpha=np.asarray(calibration['coefficients'],np.float64)[:,None,None]
    manifest=json.loads((args.assets/'manifest.json').read_text())
    assert manifest['features']==feature_names
    parts=args.out/'parts';parts.mkdir(parents=True,exist_ok=True)
    counts={}
    for pm in manifest['panels']:
        route=routes[pm['panel']]
        state=torch.load(ROOT/route['checkpoint'],map_location='cuda',weights_only=True)
        assert state['config']['features']==feature_names
        model.load_state_dict(state['state_dict'])
        alpha=np.asarray(json.loads((ROOT/route['calibration']).read_text())['coefficients'],np.float64)[:,None,None]
        name=pm['panel'];root=args.release/'corridors'/name
        assets=args.assets/name
        feature_graph=dict(np.load(assets/'graph.npz'))
        graph_np=dict(np.load(ROOT/route['model_graph']))
        graph_gpu={k:torch.from_numpy(v).cuda() for k,v in graph_np.items()}
        graph_gpu['node_id']=torch.arange(pm['offset'],pm['offset']+pm['nodes'],device='cuda')
        profile=dict(np.load(assets/'historical-profile.npz'))
        static=pd.read_csv(root/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
        lanes=static.lanes.to_numpy(np.float32)
        mapping=pd.read_csv(root/'network/ramp_attachment_map.csv')
        for split in ['validation','private']:
            files=sorted((root/split/'mainline_states_masked').rglob('*.parquet'),key=lambda f:f.name)
            ramps_by_day={p.stem[-10:].replace('_','-'):p for p in (root/split/'ramp_states').rglob('*.parquet')}
            dates=[];predictions=[];visible=[]
            for f in files:
                day=f.stem[-10:].replace('_','-');dates.append(day)
                source_frame=pd.read_parquet(f)
                x=features_for_day(source_frame,day,pm,static,profile,feature_graph,ramps_by_day[day],mapping)
                if len(feature_names)==64:
                    x=np.concatenate([x,quality_features(source_frame,pm['links'],feature_graph)],axis=0)
                tensor=torch.from_numpy(x[None].astype(np.float32)).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    raw=model(tensor,graph_gpu)[0].cpu().numpy()
                base=x[:2].astype(np.float64)*np.array([4.,2000/600])[:,None,None]
                value=np.maximum(0,base+alpha*(raw-base))
                assert np.isfinite(value).all()
                predictions.append(value);visible.append(x[6])
            prediction=np.stack(predictions);available=np.stack(visible)
            frame=pd.read_csv(args.release/'task1'/name/split/'sample_submission_state.csv',
                              dtype={k:str for k in ['panel','timestamp','station_id','link_id','mask_regime']})
            ts=pd.to_datetime(frame.timestamp,utc=True)
            di=ts.dt.strftime('%Y-%m-%d').map({d:i for i,d in enumerate(dates)}).to_numpy(int)
            ti=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
            ni=frame.link_id.map({v:i for i,v in enumerate(pm['links'])}).to_numpy(int)
            assert not available[di,ti,ni].any()
            frame['speed_kmh']=prediction[di,0,ti,ni]*25
            frame['flow_vph']=prediction[di,1,ti,ni]*600*lanes[ni]
            frame.to_csv(parts/f'state-{name}-{split}.csv',index=False,float_format='%.6f')
            counts[f'{name}/{split}']=len(frame)
            print(json.dumps(dict(panel=name,split=split,rows=len(frame))),flush=True)
    # Reuse the already validated auxiliary predictions; do not replace them
    # or the previous submission. This candidate's Task 2/4 methods are unchanged.
    for pattern in ['queue-*.csv','odme-*.csv']:
        for source in (args.previous/'parts').glob(pattern):shutil.copyfile(source,parts/source.name)
    sys.path.insert(0,str(args.previous))
    specification=importlib.util.spec_from_file_location('submission_helpers',args.previous/'build_submission.py')
    helper=importlib.util.module_from_spec(specification);specification.loader.exec_module(helper)
    helper.merge(args);helper.verify(args)
    (args.out/'candidate-scope.json').write_text(json.dumps(dict(state_model='per-corridor calibrated graph-SSM with full-day time attention',
        auxiliary_source=str(args.previous.resolve()),new_state_rows=sum(counts.values()),
        note='Experimental candidate, no upload. Task 2 and Task 4 are reused baseline outputs.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path)
    parser.add_argument('--assets',type=Path)
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--release',type=Path);parser.add_argument('--out',type=Path,default=ROOT/'candidate-submission-experts')
    parser.add_argument('--previous',type=Path,default=ROOT.parent/'IEEE-Competition-Ready')
    parser.add_argument('--route-manifest',type=Path,default=ROOT/'final-model.json')
    args=parser.parse_args()
    if args.cache:
        if args.assets is None:args.assets=ROOT/'inference-assets'
        export_assets(args.cache,args.assets)
    if not args.export_only:
        if args.release is None:parser.error('--release is required for prediction')
        if not (ROOT/'specialists-complete.json').exists():
            parser.error('Training stages are still running; wait for all-stages-complete.json before GPU inference.')
        route_info=json.loads(args.route_manifest.read_text())
        args.checkpoint=ROOT/route_info['routes'][0]['checkpoint']
        args.calibration=ROOT/route_info['routes'][0]['calibration']
        predict(args)
