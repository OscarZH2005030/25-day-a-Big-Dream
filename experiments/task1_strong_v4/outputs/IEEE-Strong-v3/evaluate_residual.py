"""Evaluate frozen flow corrections against v2 only, retaining its exact speed metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from residual_core import *


def summarize(table):
    return dict(score=float(table.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean()),
        flow_rmse=float(np.sqrt(np.dot(table.n,table.rmse_flow_per_lane**2)/table.n.sum())),
        speed_rmse=float(np.sqrt(np.dot(table.n,table.rmse_speed_kmh**2)/table.n.sum())))


def main():
    torch.set_num_threads(4)
    frozen=json.loads((ROOT/'frozen-routes.json').read_text());routes={r['panel']:r for r in frozen['routes']}
    baseline=pd.read_csv(BASE/'holdout-by-regime.csv');result=baseline.copy();details=[];checks=[]
    metadata=json.loads((CACHE/'metadata.json').read_text())
    for pm in metadata['panels']:
        panel=pm['panel'];route=routes.get(panel)
        if not route or not route['alpha']:continue
        out=ROOT/panel;state=torch.load(out/'best-model.pt',weights_only=True,map_location='cuda')
        model=ResidualFlow(pm['nodes']).cuda().eval();model.load_state_dict(state['state_dict'])
        x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
        old=Path(pm['old_cache']);y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
        base=np.load(BASE/panel/'selected-holdout.npz')['prediction']
        raw=np.load(BASE/panel/'raw-holdout.npz')['prediction']
        sums={r:np.zeros(3) for r in ['R1','R2','R3']};previous={r:np.zeros(2) for r in sums}
        selected=[];corrections=[]
        for i,d in enumerate(range(218,273)):
            m=mask[d];day=pm['dates'][d]
            b=base[i,1];z=contexts(x[d],raw[i,0],b,panel,day)
            nodes=np.broadcast_to(np.arange(pm['nodes']),m.shape)[m]
            p=infer_rows(model,z[m],nodes,b[m]);flow=b.copy()
            flow[m]=(1-route['alpha'])*b[m]+route['alpha']*p
            sq=float(np.sum((flow[m].astype(np.float64)-y[d,1][m])**2))
            osq=float(np.sum((b[m].astype(np.float64)-y[d,1][m])**2))
            regime=pm['regimes'][d];n=int(m.sum());sums[regime]+=np.array([n,sq,osq])
            selected.append(flow);corrections.append((p-b[m]).astype(np.float32))
            details.append(dict(panel=panel,day=d,regime=regime,n=n,flow_sq=sq,previous_flow_sq=osq))
        for regime,s in sums.items():
            q=float(np.sqrt(s[1]/s[0])*600);oldq=float(np.sqrt(s[2]/s[0])*600)
            row=(result.panel==panel)&(result.regime==regime);assert row.sum()==1
            assert abs(float(baseline.loc[row,'rmse_flow_per_lane'].iloc[0])-oldq)<.0001
            result.loc[row,'rmse_flow_per_lane']=q
            speed=float(result.loc[row,'rmse_speed_kmh'].iloc[0])
            result.loc[row,'S_state']=float(.54*max(0,1-speed/25)+.46*max(0,1-q/600))
        np.savez_compressed(out/'selected-holdout-flow.npz',prediction=np.stack(selected),days=np.arange(218,273))
        # Validate the checkpoint against the development prediction selected by the worker.
        source=BASE/'local-flow'/panel
        dev=infer_rows(model,np.load(out/'dev-features.npy',mmap_mode='r'),np.load(source/'dev-nodes.npy'),np.load(out/'dev-base.npy'))
        b=np.load(out/'dev-base.npy');expected=np.load(out/'selected-dev-flow.npy')
        value=(1-route['alpha'])*b+route['alpha']*dev
        diff=float(np.max(np.abs(value-expected)));assert diff<2e-6
        checks.append(dict(panel=panel,maximum_development_prediction_difference=diff))
        del model,state;torch.cuda.empty_cache()
        print(json.dumps(dict(panel=panel,phase='evaluated')),flush=True)
    result.to_csv(ROOT/'holdout-by-regime.csv',index=False)
    compared=result.merge(baseline,on=['panel','family','regime','n'],suffixes=('_new','_baseline'),validate='one_to_one')
    compared['score_delta']=compared.S_state_new-compared.S_state_baseline
    compared.to_csv(ROOT/'comparison-by-regime.csv',index=False)
    compared.groupby('panel').mean(numeric_only=True).to_csv(ROOT/'comparison-by-panel.csv')
    pd.DataFrame(details).to_csv(ROOT/'paired-daily-flow-errors.csv',index=False)
    current=summarize(result);previous=summarize(baseline)
    expected=json.loads((BASE/'evaluation-summary.json').read_text())['new']
    assert abs(previous['score']-expected['score'])<1e-12
    assert current['speed_rmse']==previous['speed_rmse']
    save(ROOT/'evaluation-summary.json',dict(status='evaluated',baseline=previous,new=current,
        score_delta=current['score']-previous['score'],flow_rmse_delta=current['flow_rmse']-previous['flow_rmse'],
        targets=int(result.n.sum()),adopted=sum(r['alpha']>0 for r in routes.values()),
        improved_directions=int((compared.groupby('panel').score_delta.mean()>0).sum()),
        unchanged_speed=True,selection='All parameters and mixing frozen on internal development before this evaluation.',
        limitation='Exploratory reuse of a previously viewed local holdout, not hidden-test evidence. No original Transformer comparison.'))
    save(ROOT/'checkpoint-parity.json',dict(passed=True,checks=checks))
    preserved=json.loads((ROOT/'preserved-v2-sha256.json').read_text())
    assert all(sha(Path(p))==v for p,v in preserved.items())
    save(ROOT/'preservation-check.json',dict(passed=True,files=len(preserved)))
    print((ROOT/'evaluation-summary.json').read_text(),flush=True)


if __name__=='__main__':main()
