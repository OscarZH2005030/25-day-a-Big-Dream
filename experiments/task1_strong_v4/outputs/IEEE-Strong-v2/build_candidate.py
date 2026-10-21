"""Infer only the newly trained branch, mixing with immutable v1 outputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch
from optimize import ROOT,V1,CACHE,WORKSPACE,save
from hybrid_model import make_model
from specialist_inference import features_for_day
from quality_features import quality_features
from local_flow_expert import FlowExpert,predict_grid


def main():
    assert (ROOT/'evaluation-summary.json').exists()
    torch.set_num_threads(4)
    release=WORKSPACE/'work/trafficflowbench-data/kaggle_public'
    previous=V1/'candidate-submission-experts'
    out=ROOT/'candidate-submission';parts=out/'parts';parts.mkdir(parents=True,exist_ok=True)
    routes={r['panel']:r for r in json.loads((ROOT/'frozen-routes.json').read_text())['routes']}
    local_routes={r['panel']:r for r in json.loads((ROOT/'final-model.json').read_text())['local_flow_routes']}
    assets=V1/'inference-assets-64'
    manifest=json.loads((assets/'manifest.json').read_text());checks=[];counts={}
    for pm in manifest['panels']:
        panel=pm['panel'];route=routes[panel];a=np.array(route['blend'])
        local=local_routes.get(panel)
        local_alpha=local['alpha'] if local else 0.
        if not route['checkpoint'] and not local_alpha:
            for split in ['validation','private']:
                source=previous/'parts'/f'state-{panel}-{split}.csv'
                shutil.copyfile(source,parts/source.name)
            continue
        folder=release/'corridors'/panel
        checkpoint_path=ROOT/panel/route['checkpoint'] if route['checkpoint'] else V1/route['previous_route']['checkpoint']
        state=torch.load(checkpoint_path,map_location='cuda',weights_only=True)
        model=make_model(state['config'],64).cuda().eval();model.load_state_dict(state['state_dict'])
        booster=None
        if local_alpha:
            local_state=torch.load(ROOT/'local-flow'/panel/'best-model.pt',map_location='cuda',weights_only=True)
            booster=FlowExpert(pm['nodes']).cuda().eval();booster.load_state_dict(local_state['state_dict'])
        feature_graph=dict(np.load(assets/panel/'graph.npz'))
        model_graph={k:torch.from_numpy(v).cuda() for k,v in np.load(V1/route['previous_route']['model_graph']).items()}
        model_graph['node_id']=torch.arange(pm['offset'],pm['offset']+pm['nodes'],device='cuda')
        static=pd.read_csv(folder/'network/fd_parameters.csv').set_index('link_id').loc[pm['links']]
        lanes=static.lanes.to_numpy(np.float32)
        profile=dict(np.load(assets/panel/'historical-profile.npz'))
        mapping=pd.read_csv(folder/'network/ramp_attachment_map.csv')
        def raw_features(file,ramp):
            frame=pd.read_parquet(file);day=file.stem[-10:].replace('_','-')
            z=features_for_day(frame,day,pm,static,profile,feature_graph,ramp,mapping)
            return np.concatenate([z,quality_features(frame,pm['links'],feature_graph)],axis=0)
        def predict(z):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                value=model(torch.from_numpy(z[None].astype(np.float32)).cuda(),model_graph)[0].float().cpu().numpy()
            assert np.isfinite(value).all();return value
        # Check raw data -> feature construction -> new checkpoint on a benchmark
        # day against the cached-input evaluation before producing new predictions.
        day='2031-01-05'
        file=next(f for f in (folder/'train/mainline_states_masked').rglob('*.parquet') if f.stem[-10:].replace('_','-')==day)
        ramp=next(f for f in (folder/'train/ramp_states').rglob('*.parquet') if f.stem[-10:].replace('_','-')==day)
        z=raw_features(file,ramp)
        assert np.array_equal(z,np.load(CACHE/panel/'features.npy',mmap_mode='r')[218])
        raw_file=ROOT/panel/'raw-holdout.npz' if route['checkpoint'] else V1/'specialists'/panel/(panel+'-holdout.npz')
        stored=np.load(raw_file)['prediction'][0]
        difference=float(np.abs(predict(z)-stored).max());assert difference<2e-6
        check=dict(panel=panel,features_bitwise_equal=True,maximum_prediction_difference=difference)
        if booster is not None:
            stored_local=np.load(ROOT/panel/'local-flow-holdout.npz')['prediction'][0]
            local_difference=float(np.abs(predict_grid(booster,z,panel,day)-stored_local).max())
            assert local_difference<2e-6;check['local_flow_prediction_difference']=local_difference
        checks.append(check)
        for split in ['validation','private']:
            target=parts/f'state-{panel}-{split}.csv'
            table=pd.read_csv(previous/'parts'/target.name,dtype={k:str for k in ['panel','timestamp','station_id','link_id','mask_regime']})
            ts=pd.to_datetime(table.timestamp,utc=True)
            dates=ts.dt.strftime('%Y-%m-%d');times=(ts.dt.hour*12+ts.dt.minute//5).to_numpy(int)
            nodes=table.link_id.map({v:i for i,v in enumerate(pm['links'])}).to_numpy(int)
            ramps={p.stem[-10:].replace('_','-'):p for p in (folder/split/'ramp_states').rglob('*.parquet')}
            files=sorted((folder/split/'mainline_states_masked').rglob('*.parquet'),key=lambda p:p.name)
            covered=np.zeros(len(table),bool)
            for file in files:
                day=file.stem[-10:].replace('_','-');selected=(dates==day).to_numpy()
                if not selected.any():continue
                z=raw_features(file,ramps[day]);prediction=predict(z)
                ti=times[selected];ni=nodes[selected]
                assert not z[6,ti,ni].any()
                new_speed=prediction[0,ti,ni]*25
                new_flow=prediction[1,ti,ni]*600*lanes[ni]
                table.loc[selected,'speed_kmh']=(1-a[0])*table.loc[selected,'speed_kmh'].to_numpy()+a[0]*new_speed
                table.loc[selected,'flow_vph']=(1-a[1])*table.loc[selected,'flow_vph'].to_numpy()+a[1]*new_flow
                if booster is not None:
                    extra=predict_grid(booster,z,panel,day)[ti,ni]*600*lanes[ni]
                    table.loc[selected,'flow_vph']=(1-local_alpha)*table.loc[selected,'flow_vph'].to_numpy()+local_alpha*extra
                covered[selected]=True
            assert covered.all()
            assert np.isfinite(table[['speed_kmh','flow_vph']].to_numpy()).all()
            table.to_csv(target,index=False,float_format='%.6f')
            counts[f'{panel}/{split}']=len(table)
            print(json.dumps(dict(panel=panel,split=split,rows=len(table))),flush=True)
        del model,state,model_graph,booster;torch.cuda.empty_cache()
    for pattern in ['queue-*.csv','odme-*.csv']:
        for source in (previous/'parts').glob(pattern):shutil.copyfile(source,parts/source.name)
    helper_path=ROOT.parent/'IEEE-Competition-Ready/build_submission.py'
    sys.path.insert(0,str(helper_path.parent))
    spec=importlib.util.spec_from_file_location('legacy_merge_helpers',helper_path)
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    args=argparse.Namespace(out=out,release=release)
    helpers.merge(args);helpers.verify(args)
    save(ROOT/'raw-input-inference-parity.json',dict(passed=True,checks=checks))
    save(out/'candidate-scope.json',dict(model='Flow-focused continuation with internal-selected channel-wise v1 fallback',
        original_submission_preserved=str(previous/'submission.csv'),new_routes=sum(bool(r['checkpoint']) for r in routes.values()),
        auxiliary='Task 2 and Task 4 retain v1 baseline predictions. No Kaggle upload.',
        inference='Mix physical outputs from current checkpoint with saved v1 per-target values; v1 CSV rounding contributes <=1e-6 physical units.'))


if __name__=='__main__':main()
