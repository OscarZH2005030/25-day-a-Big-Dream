"""Portable v4 inference. All weights and learned assets resolve beside this file.

Provide the original Kaggle kaggle_public directory separately. No training cache,
held-out labels, virtual environment, or original C-drive model folder is used.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
V1=OUT/'IEEE-Strong-v1';V2=OUT/'IEEE-Strong-v2';V3=OUT/'IEEE-Strong-v3';V4=OUT/'IEEE-Strong-v4'
for p in [OUT,V1,V2,V3,V4]:sys.path.insert(0,str(p))
from hybrid_model import make_model
from wide_model import WideGraphSSM
from specialist_inference import features_for_day
from quality_features import quality_features
import local_flow_expert as lf
import residual_core as rc


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def route_path(value):return Path(value.replace('\\','/'))


def load_model(path,wide=False):
    state=torch.load(path,map_location='cuda',weights_only=True)
    model=WideGraphSSM() if wide else make_model(state['config'],64)
    model=model.cuda().eval();model.load_state_dict(state['state_dict']);return model


def forward(model,x,graph):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        return model(torch.from_numpy(x[None].astype(np.float32)).cuda(),graph)[0].float().cpu().numpy()


def rounded(value):return np.round(np.asarray(value,np.float64),6)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Path containing submission_key.csv and corridors/')
    parser.add_argument('--out',type=Path,default=ROOT/'new-predictions')
    parser.add_argument('--panel',action='append',help='Repeat to select panels; default all ten')
    parser.add_argument('--split',choices=['validation','private'],action='append')
    parser.add_argument('--day',help='Optional YYYY-MM-DD for a small inference check')
    parser.add_argument('--smoke-reference',action='store_true',help='Rebuild March1 for all ten roads and compare included reference predictions')
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch is required; see README.md')
    torch.set_num_threads(4)
    if args.smoke_reference:args.day='2031-03-01';args.split=['validation']
    assert args.data.joinpath('submission_key.csv').exists(),'--data must point to kaggle_public/'
    parts=args.out/'parts';parts.mkdir(parents=True,exist_ok=True)
    manifest=read(V1/'inference-assets-64/manifest.json')
    r1={r['panel']:r for r in read(V1/'final-model.json')['routes']}
    second=read(V2/'final-model.json');r2={r['panel']:r for r in second['routes']}
    local={r['panel']:r for r in second['local_flow_routes'] if r['alpha']}
    third={r['panel']:r for r in read(V3/'frozen-routes.json')['routes'] if r['alpha']}
    fourth={r['panel']:r for r in read(V4/'frozen-routes.json')['routes'] if r['alpha']}
    checks=[]
    for pm in manifest['panels']:
        panel=pm['panel']
        if args.panel and panel not in args.panel:continue
        first=r1[panel];second=r2[panel];assets=V1/'inference-assets-64'/panel
        feature_graph=dict(np.load(assets/'graph.npz'));profile=dict(np.load(assets/'historical-profile.npz'))
        graph={k:torch.from_numpy(v).cuda() for k,v in np.load(V1/route_path(first['model_graph'])).items()}
        graph['node_id']=torch.arange(pm['offset'],pm['offset']+pm['nodes'],device='cuda')
        folder=args.data/'corridors'/panel
        static=pd.read_csv(folder/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
        lane32=static.lanes.to_numpy(np.float32);lane64=static.lanes.to_numpy(np.float64)
        mapping=pd.read_csv(folder/'network/ramp_attachment_map.csv')
        first_model=load_model(V1/route_path(first['checkpoint']))
        second_model=load_model(V2/panel/second['checkpoint']) if second['checkpoint'] else first_model
        calibration=np.array(read(V1/route_path(first['calibration']))['coefficients'])[:,None,None]
        booster=None;corrector=None;wide=None
        if panel in local:
            booster=lf.FlowExpert(pm['nodes']).cuda().eval()
            booster.load_state_dict(torch.load(V2/'local-flow'/panel/'best-model.pt',map_location='cuda',weights_only=True)['state_dict'])
        if panel in third:
            corrector=rc.ResidualFlow(pm['nodes']).cuda().eval()
            corrector.load_state_dict(torch.load(V3/panel/'best-model.pt',map_location='cuda',weights_only=True)['state_dict'])
        if panel in fourth:wide=load_model(V4/panel/'best-model.pt',wide=True)
        for split in args.split or ['validation','private']:
            filename=f'state-{panel}-{split}.csv'
            frame=pd.read_csv(args.data/'task1'/panel/split/'sample_submission_state.csv',dtype={k:str for k in ['panel','timestamp','station_id','link_id','mask_regime']})
            for col in ['speed_kmh','flow_vph']:frame[col]=frame[col].astype(np.float64)
            ts=pd.to_datetime(frame.timestamp,utc=True)
            dates=ts.dt.strftime('%Y-%m-%d');times=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
            nodes=frame.link_id.map({v:i for i,v in enumerate(pm['links'])}).to_numpy(int)
            selected_all=np.zeros(len(frame),bool)
            ramps={p.stem[-10:].replace('_','-'):p for p in (folder/split/'ramp_states').rglob('*.parquet')}
            for file in sorted((folder/split/'mainline_states_masked').rglob('*.parquet'),key=lambda p:p.name):
                day=file.stem[-10:].replace('_','-')
                if args.day and day!=args.day:continue
                sl=(dates==day).to_numpy()
                if not sl.any():continue
                source=pd.read_parquet(file)
                x=np.concatenate([features_for_day(source,day,pm,static,profile,feature_graph,ramps[day],mapping),quality_features(source,pm['links'],feature_graph)],0)
                ti=times[sl];ni=nodes[sl];assert not x[6,ti,ni].any()
                raw1=forward(first_model,x,graph)
                base=x[:2].astype(np.float64)*np.array([4.,2000/600])[:,None,None]
                old=np.maximum(0,base+calibration*(raw1-base))
                speed=rounded(old[0,ti,ni]*25);flow=rounded(old[1,ti,ni]*600*lane32[ni])
                raw2=forward(second_model,x,graph) if second['checkpoint'] else raw1
                a=second['blend']
                speed=rounded((1-a[0])*speed+a[0]*(raw2[0,ti,ni]*25))
                flow=(1-a[1])*flow+a[1]*(raw2[1,ti,ni]*600*lane32[ni])
                if booster is not None:
                    extra=lf.predict_grid(booster,x,panel,day)[ti,ni]*600*lane32[ni];alpha=local[panel]['alpha']
                    flow=(1-alpha)*flow+alpha*extra
                flow=rounded(flow)
                if corrector is not None:
                    b=(flow/(600*lane64[ni])).astype(np.float32)
                    ctx=rc.contexts(x,raw2[0],raw2[1],panel,day)[ti,ni];ctx[:,135]=b.astype(np.float16)
                    corrected=rc.infer_rows(corrector,ctx,ni,b)
                    flow=rounded(np.maximum(0,flow+third[panel]['alpha']*((corrected-b)*600*lane64[ni])))
                if wide is not None:
                    big=forward(wide,x,graph)
                    delta=(big[1,ti,ni]-raw2[1,ti,ni])*600*lane64[ni]*fourth[panel]['alpha']
                    flow=rounded(np.maximum(0,flow+delta))
                frame.loc[sl,'speed_kmh']=speed;frame.loc[sl,'flow_vph']=flow;selected_all[sl]=True
            frame=frame.loc[selected_all].copy();assert len(frame)>0
            assert np.isfinite(frame[['speed_kmh','flow_vph']].to_numpy()).all()
            frame.to_csv(parts/filename,index=False,float_format='%.6f')
            if args.smoke_reference:
                ref=pd.read_csv(ROOT/'reference-predictions'/filename,dtype={k:str for k in ['panel','timestamp','station_id','link_id','mask_regime']})
                ref=ref.loc[pd.to_datetime(ref.timestamp,utc=True).dt.strftime('%Y-%m-%d')==args.day]
                assert frame[['timestamp','link_id']].reset_index(drop=True).equals(ref[['timestamp','link_id']].reset_index(drop=True))
                difference=np.abs(frame[['speed_kmh','flow_vph']].to_numpy()-ref[['speed_kmh','flow_vph']].to_numpy()).max(0)
                assert difference[0]<.0001 and difference[1]<.01,(panel,difference)
                checks.append(dict(panel=panel,rows=len(frame),maximum_speed_difference_kmh=float(difference[0]),maximum_flow_difference_vph=float(difference[1])))
            print(json.dumps(dict(panel=panel,split=split,rows=len(frame))),flush=True)
        del first_model,second_model,booster,corrector,wide,graph;torch.cuda.empty_cache()
    if args.smoke_reference:
        (args.out/'portable-inference-check.json').write_text(json.dumps(dict(passed=True,day=args.day,checks=checks,model_root=str(ROOT),external_training_cache_used=False),indent=2));return
    if not args.panel and not args.day and not args.split:
        for pattern in ['queue-*.csv','odme-*.csv']:
            for p in (ROOT/'reference-predictions').glob(pattern):shutil.copyfile(p,parts/p.name)
        helperpath=OUT/'IEEE-Competition-Ready/build_submission.py';sys.path.insert(0,str(helperpath.parent))
        spec=importlib.util.spec_from_file_location('portable_merge',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        parameters=argparse.Namespace(out=args.out,release=args.data);helper.merge(parameters);helper.verify(parameters)
    else:print('Partial state inference only: merge is skipped.',flush=True)


if __name__=='__main__':main()
