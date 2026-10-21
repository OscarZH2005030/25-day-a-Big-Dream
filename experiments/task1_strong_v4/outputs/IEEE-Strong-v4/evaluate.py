"""Evaluate only the candidates frozen on the separate development interval."""
import json
import numpy as np
import pandas as pd
import torch
from common import *


def summary(t):
    return dict(score=float(t.groupby(['family','panel']).S_state.mean().groupby('family').mean().mean()),
        flow_rmse=float(np.sqrt(np.dot(t.n,t.rmse_flow_per_lane**2)/t.n.sum())),
        speed_rmse=float(np.sqrt(np.dot(t.n,t.rmse_speed_kmh**2)/t.n.sum())))


def main():
    torch.set_num_threads(4)
    routes=json.loads((ROOT/'frozen-routes.json').read_text())['routes']
    before=pd.read_csv(V3/'holdout-by-regime.csv');after=before.copy();checks=[];daily=[]
    for route in routes:
        if not route['alpha']:continue
        panel=route['panel'];pm=metadata(panel);out=ROOT/panel
        model=WideGraphSSM().cuda().eval()
        model.load_state_dict(torch.load(out/'best-model.pt',map_location='cuda',weights_only=True)['state_dict'])
        narrow,graph=narrow_model(pm);del narrow
        old=Path(pm['old_cache']);mask=np.load(old/'target.npy',mmap_mode='r');y=np.load(old/'y.npy',mmap_mode='r')
        x=np.load(CACHE/panel/'features.npy',mmap_mode='r')
        raw_dev=np.stack([predict(model,x[d],graph)[1] for d in range(196,218)])
        current=current_dev(pm,mask);narrow_dev=np.load(out/'narrow-dev-flow.npy')
        selected=np.maximum(0,current[:,1]+route['alpha']*(raw_dev-narrow_dev))
        difference=float(np.max(np.abs(selected-np.load(out/'selected-dev-flow.npy'))));assert difference<2e-6
        checks.append(dict(panel=panel,development_prediction_maximum_difference=difference))
        current=np.load(V3/panel/'selected-holdout-flow.npz')['prediction']
        narrow=np.load(BASE/panel/'raw-holdout.npz')['prediction'][:,1]
        selected=[];raws=[];sums={r:np.zeros(3) for r in ['R1','R2','R3']}
        for i,d in enumerate(range(218,273)):
            raw=predict(model,x[d],graph)[1];flow=np.maximum(0,current[i]+route['alpha']*(raw-narrow[i]))
            m=mask[d];n=int(m.sum());sq=float(((flow[m].astype(np.float64)-y[d,1][m])**2).sum())
            previous=float(((current[i][m].astype(np.float64)-y[d,1][m])**2).sum())
            sums[pm['regimes'][d]]+=[n,sq,previous];selected.append(flow);raws.append(raw)
            daily.append(dict(panel=panel,day=d,regime=pm['regimes'][d],n=n,flow_sq=sq,previous_flow_sq=previous))
        for regime,(n,sq,psq) in sums.items():
            row=(after.panel==panel)&(after.regime==regime);assert row.sum()==1
            assert abs(float(before.loc[row,'rmse_flow_per_lane'].iloc[0])-np.sqrt(psq/n)*600)<1e-4
            q=float(np.sqrt(sq/n)*600);speed=float(after.loc[row,'rmse_speed_kmh'].iloc[0])
            after.loc[row,'rmse_flow_per_lane']=q;after.loc[row,'S_state']=.54*max(0,1-speed/25)+.46*max(0,1-q/600)
        np.savez_compressed(out/'selected-holdout-flow.npz',prediction=np.stack(selected),days=np.arange(218,273))
        np.savez_compressed(out/'raw-holdout-flow.npz',prediction=np.stack(raws),days=np.arange(218,273))
        del model;torch.cuda.empty_cache();print(json.dumps(dict(panel=panel,phase='evaluated')),flush=True)
    after.to_csv(ROOT/'holdout-by-regime.csv',index=False)
    compared=after.merge(before,on=['panel','family','regime','n'],suffixes=('_new','_baseline'),validate='one_to_one')
    compared['score_delta']=compared.S_state_new-compared.S_state_baseline
    compared.to_csv(ROOT/'comparison-by-regime.csv',index=False)
    compared.groupby('panel').mean(numeric_only=True).to_csv(ROOT/'comparison-by-panel.csv')
    pd.DataFrame(daily).to_csv(ROOT/'paired-daily-flow-errors.csv',index=False)
    a=summary(after);b=summary(before);assert a['speed_rmse']==b['speed_rmse']
    assert abs(b['score']-json.loads((V3/'evaluation-summary.json').read_text())['new']['score'])<1e-12
    save(ROOT/'evaluation-summary.json',dict(baseline=b,new=a,score_delta=a['score']-b['score'],
        flow_rmse_delta=a['flow_rmse']-b['flow_rmse'],targets=int(after.n.sum()),
        adopted=sum(r['alpha']>0 for r in routes),improved_directions=int((compared.groupby('panel').score_delta.mean()>0).sum()),
        unchanged_speed=True,holdout_previously_seen=True,hidden_test_evidence=False))
    save(ROOT/'checkpoint-parity.json',dict(passed=True,checks=checks))
    print((ROOT/'evaluation-summary.json').read_text(),flush=True)


if __name__=='__main__':main()
