"""Apply flow corrections to the v2 submission; preserve all other values."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from residual_core import *
from specialist_inference import features_for_day
from quality_features import quality_features


def main():
    assert (ROOT/'evaluation-summary.json').exists()
    torch.set_num_threads(4)
    routes={r['panel']:r for r in json.loads((ROOT/'frozen-routes.json').read_text())['routes'] if r['alpha']>0}
    if not routes:
        save(ROOT/'candidate-reuse.json',dict(submission=str(BASE/'candidate-submission/submission.csv'),reason='All correction coefficients are zero; keep the existing validated candidate.'))
        return
    release=WORKSPACE/'work/trafficflowbench-data/kaggle_public'
    old=BASE/'candidate-submission';out=ROOT/'candidate-submission';parts=out/'parts';parts.mkdir(parents=True,exist_ok=True)
    asset_root=V1/'inference-assets-64';manifest=json.loads((asset_root/'manifest.json').read_text());checks=[]
    for pm in manifest['panels']:
        panel=pm['panel'];route=routes.get(panel)
        if not route:
            for split in ['validation','private']:
                p=old/'parts'/f'state-{panel}-{split}.csv';shutil.copyfile(p,parts/p.name)
            continue
        state=torch.load(ROOT/panel/'best-model.pt',map_location='cuda',weights_only=True)
        correction=ResidualFlow(pm['nodes']).cuda().eval();correction.load_state_dict(state['state_dict'])
        nn_model,g,unused,unused_alpha=base_model(pm)
        del unused
        folder=release/'corridors'/panel
        feature_graph=dict(np.load(asset_root/panel/'graph.npz'))
        profile=dict(np.load(asset_root/panel/'historical-profile.npz'))
        static=pd.read_csv(folder/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
        lanes=static.lanes.to_numpy(np.float64);mapping=pd.read_csv(folder/'network/ramp_attachment_map.csv')
        def raw_features(file,ramp):
            frame=pd.read_parquet(file);day=file.stem[-10:].replace('_','-')
            z=features_for_day(frame,day,pm,static,profile,feature_graph,ramp,mapping)
            return np.concatenate([z,quality_features(frame,pm['links'],feature_graph)],axis=0)
        def neural(z):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                return nn_model(torch.from_numpy(z[None].astype(np.float32)).cuda(),g)[0].float().cpu().numpy()
        # Raw-file/cache parity for the new correction pipeline on a benchmark day.
        day='2031-01-05'
        file=next(f for f in (folder/'train/mainline_states_masked').rglob('*.parquet') if f.stem[-10:].replace('_','-')==day)
        ramp=next(f for f in (folder/'train/ramp_states').rglob('*.parquet') if f.stem[-10:].replace('_','-')==day)
        z=raw_features(file,ramp);cached=np.load(CACHE/panel/'features.npy',mmap_mode='r')[218]
        assert np.array_equal(z,cached)
        raw=neural(z);stored=np.load(BASE/panel/'raw-holdout.npz')['prediction'][0]
        difference=float(np.max(np.abs(raw-stored)));assert difference<2e-6
        b=np.load(BASE/panel/'selected-holdout.npz')['prediction'][0,1]
        old_cache=Path(next(p for p in json.loads((CACHE/'metadata.json').read_text())['panels'] if p['panel']==panel)['old_cache'])
        m=np.load(old_cache/'target.npy',mmap_mode='r')[218];nodes=np.broadcast_to(np.arange(pm['nodes']),m.shape)[m]
        ctx=contexts(z,raw[0],b,panel,day);p=infer_rows(correction,ctx[m],nodes,b[m]);value=(1-route['alpha'])*b[m]+route['alpha']*p
        expected=np.load(ROOT/panel/'selected-holdout-flow.npz')['prediction'][0][m]
        corr_diff=float(np.max(np.abs(value-expected)));assert corr_diff<2e-6
        checks.append(dict(panel=panel,features_bitwise_equal=True,neural_prediction_difference=difference,corrected_flow_difference=corr_diff))
        for split in ['validation','private']:
            file_out=parts/f'state-{panel}-{split}.csv'
            table=pd.read_csv(old/'parts'/file_out.name,dtype={k:str for k in ['panel','timestamp','station_id','link_id','mask_regime']})
            original_speed=table.speed_kmh.to_numpy().copy()
            ts=pd.to_datetime(table.timestamp,utc=True);days=ts.dt.strftime('%Y-%m-%d')
            times=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int);nodes=table.link_id.map({v:i for i,v in enumerate(pm['links'])}).to_numpy(int)
            ramps={p.stem[-10:].replace('_','-'):p for p in (folder/split/'ramp_states').rglob('*.parquet')}
            covered=np.zeros(len(table),bool)
            for file in sorted((folder/split/'mainline_states_masked').rglob('*.parquet'),key=lambda p:p.name):
                day=file.stem[-10:].replace('_','-');selected=(days==day).to_numpy()
                if not selected.any():continue
                z=raw_features(file,ramps[day]);raw=neural(z);ti=times[selected];ni=nodes[selected]
                assert not z[6,ti,ni].any()
                base=(table.loc[selected,'flow_vph'].to_numpy()/(600*lanes[ni])).astype(np.float32)
                # Build local context with the immutable v2 full-day flow. Only the
                # two appended neural channels depend on this flow, so replace
                # the target values after creating the context grid.
                ctx=contexts(z,raw[0],raw[1],panel,day)[ti,ni]
                ctx[:,135]=base.astype(np.float16)
                p=infer_rows(correction,ctx,ni,base)
                delta=(p-base)*600*lanes[ni]
                table.loc[selected,'flow_vph']=np.maximum(0,table.loc[selected,'flow_vph'].to_numpy()+route['alpha']*delta)
                covered[selected]=True
            assert covered.all() and np.array_equal(table.speed_kmh.to_numpy(),original_speed)
            table.to_csv(file_out,index=False,float_format='%.6f')
            print(json.dumps(dict(panel=panel,split=split,rows=len(table),speed_unchanged=True)),flush=True)
        del correction,nn_model,g,state;torch.cuda.empty_cache()
    for pattern in ['queue-*.csv','odme-*.csv']:
        for source in (old/'parts').glob(pattern):shutil.copyfile(source,parts/source.name)
    helper_path=ROOT.parent/'IEEE-Competition-Ready/build_submission.py';sys.path.insert(0,str(helper_path.parent))
    spec=importlib.util.spec_from_file_location('merge_v3',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    args=argparse.Namespace(out=out,release=release);helper.merge(args);helper.verify(args)
    save(ROOT/'raw-inference-parity.json',dict(passed=True,checks=checks))
    save(out/'scope.json',dict(baseline=str(BASE/'candidate-submission/submission.csv'),changed='Only flow values on internally selected roads',speed_unchanged=True,task2_task4_unchanged=True,kaggle_uploaded=False))


if __name__=='__main__':main()
