"""Evaluate frozen v2 routes once, with paired daily comparison to v1."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from optimize import ROOT,V1,CACHE,sha,save
from hybrid_model import make_model
from local_flow_expert import FlowExpert,predict_grid


def main():
    assert (ROOT/'frozen-routes.json').exists(), 'Freeze all internal selections first'
    assert (ROOT/'final-model.json').exists(), 'Freeze the optional local experts first'
    torch.set_num_threads(4)
    metadata=json.loads((CACHE/'metadata.json').read_text())
    routes={v['panel']:v for v in json.loads((ROOT/'frozen-routes.json').read_text())['routes']}
    local_routes={v['panel']:v for v in json.loads((ROOT/'final-model.json').read_text())['local_flow_routes']}
    all_rows=[];old_rows=[];daily=[];dev=[];offset=0;parity=[]
    for pm in metadata['panels']:
        panel=pm['panel'];route=routes[panel];old_route=route['previous_route'];out=ROOT/panel
        old=Path(pm['old_cache']);y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
        x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
        alpha=np.array(json.loads((V1/old_route['calibration']).read_text())['coefficients'])[:,None,None]
        with np.load(V1/'specialists'/panel/(panel+'-holdout.npz')) as f:
            assert np.array_equal(f['days'],np.arange(218,273));old_raw=f['prediction']
        previous=[]
        for i,d in enumerate(range(218,273)):
            base=np.array(x[d,:2],np.float64)*np.array([4.,2000/600])[:,None,None]
            previous.append(np.maximum(0,base+alpha*(old_raw[i]-base)))
        previous=np.stack(previous)
        if route['checkpoint']:
            state=torch.load(out/route['checkpoint'],map_location='cuda',weights_only=True)
            model=make_model(state['config'],len(metadata['features'])).cuda().eval();model.load_state_dict(state['state_dict'])
            g={k:torch.from_numpy(v).cuda() for k,v in np.load(V1/old_route['model_graph']).items()}
            g['node_id']=torch.arange(offset,offset+pm['nodes'],device='cuda')
            raw=[]
            for d in range(218,273):
                z=torch.from_numpy(np.array(x[d:d+1],np.float32)).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=model(z,g)[0].float().cpu().numpy()
                assert np.isfinite(p).all();raw.append(p)
            raw=np.stack(raw)
            np.savez_compressed(out/'raw-holdout.npz',prediction=raw,days=np.arange(218,273))
            a=np.array(route['blend'])[None,:,None,None]
            selected=(1-a)*previous+a*raw
            # Recheck exact selected development inference independently.
            check=[]
            for d in range(196,218):
                z=torch.from_numpy(np.array(x[d:d+1],np.float32)).cuda()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):p=model(z,g)[0].float().cpu().numpy()
                check.append(p)
            olddev=np.load(out/'previous-development.npz')['prediction']
            expected=np.load(out/'selected-development.npz')['prediction']
            actual=(1-a)*olddev+a*np.stack(check)
            difference=float(np.max(np.abs(actual-expected)))
            assert difference<2e-6
            parity.append(dict(panel=panel,maximum_difference=difference))
            del model,g,state;torch.cuda.empty_cache()
        else:selected=previous
        local=local_routes.get(panel)
        if local and local['alpha']>0:
            state=torch.load(ROOT/'local-flow'/panel/'best-model.pt',map_location='cuda',weights_only=True)
            booster=FlowExpert(pm['nodes']).cuda().eval();booster.load_state_dict(state['state_dict'])
            extra=np.stack([predict_grid(booster,np.array(x[d]),panel,pm['dates'][d]) for d in range(218,273)])
            selected[:,1]=(1-local['alpha'])*selected[:,1]+local['alpha']*extra
            np.savez_compressed(out/'local-flow-holdout.npz',prediction=extra,days=np.arange(218,273))
            del booster,state;torch.cuda.empty_cache()
        offset+=pm['nodes']
        np.savez_compressed(out/'selected-holdout.npz',prediction=selected.astype(np.float32),days=np.arange(218,273))
        storage={r:np.zeros(3) for r in ['R1','R2','R3']};old_storage={r:np.zeros(3) for r in storage}
        for i,d in enumerate(range(218,273)):
            m=mask[d];truth=y[d];n=int(m.sum())
            sq=((selected[i]-truth)[:,m]**2).sum(1)
            osq=((previous[i]-truth)[:,m]**2).sum(1)
            regime=pm['regimes'][d];storage[regime]+=np.r_[n,sq];old_storage[regime]+=np.r_[n,osq]
            daily.append(dict(panel=panel,family=pm['family'],regime=regime,day=d,n=n,
                              speed_sq=sq[0],flow_sq=sq[1],old_speed_sq=osq[0],old_flow_sq=osq[1]))
        for source,destination in [(storage,all_rows),(old_storage,old_rows)]:
            for r,s in source.items():
                n=s[0];rmse=np.sqrt(s[1:]/n)
                destination.append(dict(panel=panel,family=pm['family'],regime=r,n=int(n),
                    rmse_speed_kmh=rmse[0]*25,rmse_flow_per_lane=rmse[1]*600,
                    S_state=float(np.dot([.54,.46],np.maximum(0,1-rmse)))))
        final_dev_score=route['metrics']['score'];final_dev_flow=route['metrics']['pooled'][1]
        if local and local['alpha']>0:
            final_dev_score+=.46*(route['metrics']['macro'][1]-local['macro_flow'])/600
            final_dev_flow=local['pooled_flow']
        dev.append(dict(panel=panel,previous_score=json.loads((out/'previous-development-metrics.json').read_text())['score'],
                        new_score=final_dev_score,old_flow=json.loads((out/'previous-development-metrics.json').read_text())['pooled'][1],
                        new_flow=final_dev_flow,epoch=route['epoch'],speed_blend=route['blend'][0],flow_blend=route['blend'][1],
                        local_flow_blend=local['alpha'] if local else 0.))
        print(json.dumps(dict(panel=panel,phase='holdout_evaluated')),flush=True)
    current=pd.DataFrame(all_rows);previous=pd.DataFrame(old_rows)
    current.to_csv(ROOT/'holdout-by-regime.csv',index=False);previous.to_csv(ROOT/'previous-holdout-by-regime.csv',index=False)
    reference=pd.read_csv(V1/'references/transformer-all-metrics.csv');reference=reference[reference.regime.isin(['R1','R2','R3'])]
    reference['family']=reference.panel.map({p['panel']:p['family'] for p in metadata['panels']})
    archived=pd.read_csv(V1/'specialists/calibrated-holdout-by-regime.csv')
    matched=previous.merge(archived,on=['panel','regime'],suffixes=('_recomputed','_archived'),validate='one_to_one')
    assert np.array_equal(matched.n_recomputed,matched.n_archived)
    assert np.max(np.abs(matched.S_state_recomputed-matched.S_state_archived))<2e-8
    paired=current.merge(previous,on=['panel','family','regime','n'],suffixes=('_new','_previous'),validate='one_to_one')
    paired=paired.merge(reference[['panel','regime','S_state','rmse_speed_kmh','rmse_flow_per_lane']],on=['panel','regime'],validate='one_to_one')
    paired['delta_vs_previous']=paired.S_state_new-paired.S_state_previous
    paired['delta_vs_transformer']=paired.S_state_new-paired.S_state
    paired.to_csv(ROOT/'comparison-by-regime.csv',index=False)
    by=paired.groupby('panel').mean(numeric_only=True)
    by.to_csv(ROOT/'comparison-by-panel.csv')
    pd.DataFrame(dev).to_csv(ROOT/'internal-selection.csv',index=False)
    daily=pd.DataFrame(daily);daily.to_csv(ROOT/'paired-daily-errors.csv',index=False)
    def summarize(frame):
        return dict(score=float(frame.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean()),
            speed_rmse=float(np.sqrt(np.dot(frame.n,frame.rmse_speed_kmh**2)/frame.n.sum())),
            flow_rmse=float(np.sqrt(np.dot(frame.n,frame.rmse_flow_per_lane**2)/frame.n.sum())))
    result=dict(status='evaluated',previous=summarize(previous),new=summarize(current),transformer=summarize(reference),
        targets=int(current.n.sum()),directions_improved_vs_previous=int((by.delta_vs_previous>0).sum()),
        directions_improved_vs_transformer=int((by.delta_vs_transformer>0).sum()),
        adapted_routes=sum(bool(v['checkpoint']) for v in routes.values()),
        local_flow_routes=sum(v['alpha']>0 for v in local_routes.values()),
        selection='Frozen internal-only selections; this evaluation does not choose or change routes.',
        limitation='Exploratory reuse of a historically observed holdout. Not hidden-test evidence.')
    assert result['targets']==6636705
    save(ROOT/'evaluation-summary.json',result)
    save(ROOT/'checkpoint-inference-parity.json',dict(passed=True,checks=parity))
    # Day-block paired bootstrap: resample the same dates for all corridors.
    # Descriptive uncertainty conditional on the selected models, not a correction
    # for historical reuse or hyperparameter search.
    arr=np.zeros((55,10,3,5));panels=list(routes)
    for row in daily.itertuples():
        arr[row.day-218,panels.index(row.panel),['R1','R2','R3'].index(row.regime)]=[row.n,row.speed_sq,row.flow_sq,row.old_speed_sq,row.old_flow_sq]
    rng=np.random.default_rng(20261024);differences=[];flow_differences=[]
    for _ in range(2000):
        s=arr[rng.integers(55,size=55)].sum(0);counts=s[...,0]
        if (counts==0).any():continue
        a=np.sqrt(s[...,1:3]/counts[...,None]);b=np.sqrt(s[...,3:5]/counts[...,None])
        differences.append(float((np.maximum(0,1-a)*[.54,.46]).sum(-1).mean()-(np.maximum(0,1-b)*[.54,.46]).sum(-1).mean()))
        flow_differences.append(float((np.sqrt(s[...,2].sum()/counts.sum())-np.sqrt(s[...,4].sum()/counts.sum()))*600))
    save(ROOT/'paired-bootstrap.json',dict(replicates=len(differences),score_delta_95pct=np.quantile(differences,[.025,.975]).tolist(),
        flow_rmse_delta_95pct=np.quantile(flow_differences,[.025,.975]).tolist(),
        limitation='Descriptive paired date-block bootstrap conditional on selected models. Cannot remove selection bias from historical holdout reuse.'))
    preserved=json.loads((ROOT/'preserved-v1-sha256.json').read_text())
    assert all(sha(Path(p))==v for p,v in preserved.items())
    save(ROOT/'preservation-check.json',dict(passed=True,files=len(preserved)))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
