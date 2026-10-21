"""Apply only the internally accepted larger-model changes to immutable v3."""
import argparse
import importlib.util
import json
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from common import *
from specialist_inference import features_for_day
from quality_features import quality_features


def main():
    torch.set_num_threads(4)
    routes={r['panel']:r for r in json.loads((ROOT/'frozen-routes.json').read_text())['routes'] if r['alpha']>0}
    if not routes:
        save(ROOT/'candidate-reuse.json',dict(submission=str(V3/'candidate-submission/submission.csv'),reason='No larger model passed internal selection; keep v3.'));return
    release=WORKSPACE/'work/trafficflowbench-data/kaggle_public'
    old=V3/'candidate-submission';out=ROOT/'candidate-submission';parts=out/'parts';parts.mkdir(parents=True,exist_ok=True)
    assets=V1/'inference-assets-64';manifest=json.loads((assets/'manifest.json').read_text());checks=[]
    for pm in manifest['panels']:
        panel=pm['panel'];route=routes.get(panel)
        if not route:
            for split in ['validation','private']:
                p=old/'parts'/f'state-{panel}-{split}.csv';shutil.copyfile(p,parts/p.name)
            continue
        wide=WideGraphSSM().cuda().eval();wide.load_state_dict(torch.load(ROOT/panel/'best-model.pt',map_location='cuda',weights_only=True)['state_dict'])
        narrow,graph=narrow_model(pm);folder=release/'corridors'/panel
        feature_graph=dict(np.load(assets/panel/'graph.npz'));profile=dict(np.load(assets/panel/'historical-profile.npz'))
        static=pd.read_csv(folder/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
        lanes=static.lanes.to_numpy(np.float64);mapping=pd.read_csv(folder/'network/ramp_attachment_map.csv')
        def features(file,ramp):
            frame=pd.read_parquet(file);day=file.stem[-10:].replace('_','-')
            z=features_for_day(frame,day,pm,static,profile,feature_graph,ramp,mapping)
            return np.concatenate([z,quality_features(frame,pm['links'],feature_graph)],axis=0)
        day='2031-01-05'
        f=next(f for f in (folder/'train/mainline_states_masked').rglob('*.parquet') if f.stem[-10:].replace('_','-')==day)
        ramp=next(f for f in (folder/'train/ramp_states').rglob('*.parquet') if f.stem[-10:].replace('_','-')==day)
        z=features(f,ramp);cached=np.load(CACHE/panel/'features.npy',mmap_mode='r')[218];assert np.array_equal(z,cached)
        p=predict(wide,z,graph)[1];q=predict(narrow,z,graph)[1]
        stored=np.load(ROOT/panel/'raw-holdout-flow.npz')['prediction'][0]
        wide_diff=float(np.max(np.abs(p-stored)));assert wide_diff<2e-6
        storedq=np.load(BASE/panel/'raw-holdout.npz')['prediction'][0,1]
        narrow_diff=float(np.max(np.abs(q-storedq)));assert narrow_diff<2e-6
        current=np.load(V3/panel/'selected-holdout-flow.npz')['prediction'][0]
        value=np.maximum(0,current+route['alpha']*(p-q))
        expected=np.load(ROOT/panel/'selected-holdout-flow.npz')['prediction'][0]
        diff=float(np.max(np.abs(value-expected)));assert diff<2e-6
        checks.append(dict(panel=panel,features_bitwise_equal=True,wide_difference=wide_diff,narrow_difference=narrow_diff,final_flow_difference=diff))
        for split in ['validation','private']:
            dest=parts/f'state-{panel}-{split}.csv'
            table=pd.read_csv(old/'parts'/dest.name,dtype={k:str for k in ['panel','timestamp','station_id','link_id','mask_regime']})
            speed=table.speed_kmh.to_numpy().copy();ts=pd.to_datetime(table.timestamp,utc=True)
            days=ts.dt.strftime('%Y-%m-%d');times=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
            nodes=table.link_id.map({v:i for i,v in enumerate(pm['links'])}).to_numpy(int)
            ramps={p.stem[-10:].replace('_','-'):p for p in (folder/split/'ramp_states').rglob('*.parquet')};covered=np.zeros(len(table),bool)
            for file in sorted((folder/split/'mainline_states_masked').rglob('*.parquet'),key=lambda p:p.name):
                day=file.stem[-10:].replace('_','-');sl=(days==day).to_numpy()
                if not sl.any():continue
                z=features(file,ramps[day]);a=predict(wide,z,graph)[1];b=predict(narrow,z,graph)[1];ti=times[sl];ni=nodes[sl]
                assert not z[6,ti,ni].any()
                delta=(a[ti,ni]-b[ti,ni])*600*lanes[ni]*route['alpha']
                table.loc[sl,'flow_vph']=np.maximum(0,table.loc[sl,'flow_vph'].to_numpy()+delta);covered[sl]=True
            assert covered.all() and np.array_equal(speed,table.speed_kmh.to_numpy())
            table.to_csv(dest,index=False,float_format='%.6f');print(json.dumps(dict(panel=panel,split=split,rows=len(table))),flush=True)
        del wide,narrow,graph;torch.cuda.empty_cache()
    for pattern in ['queue-*.csv','odme-*.csv']:
        for p in (old/'parts').glob(pattern):shutil.copyfile(p,parts/p.name)
    source=ROOT.parent/'IEEE-Competition-Ready/build_submission.py';sys.path.insert(0,str(source.parent))
    spec=importlib.util.spec_from_file_location('merge_v4',source);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    args=argparse.Namespace(out=out,release=release);helper.merge(args);helper.verify(args)
    save(ROOT/'raw-inference-parity.json',dict(passed=True,checks=checks))


if __name__=='__main__':main()
