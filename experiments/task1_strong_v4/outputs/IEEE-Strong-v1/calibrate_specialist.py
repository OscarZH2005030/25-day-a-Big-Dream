"""Pre-specified two-scalar correction calibration on INTERNAL development only.

This file is prepared before the new training run's holdout results exist.
Bounds [0,2] and one coefficient per output are fixed here. Raw and calibrated
holdout results are both reported; holdout is never used to choose coefficients.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
from hybrid_model import make_model


def score(rows,column='S_state'):
    frame=pd.DataFrame(rows)
    return float(frame.groupby(['family','panel'])[column].mean().groupby('family').mean().mean())


def metric_rows(pm,storage):
    rows=[]
    for regime,values in storage.items():
        n=values[0];rmse=np.sqrt(values[1:]/n)
        rows.append(dict(panel=pm['panel'],family=pm['family'],regime=regime,n=int(n),
                         rmse_speed_kmh=float(rmse[0]*25),rmse_flow_per_lane=float(rmse[1]*600),
                         S_state=float(np.dot([.54,.46],np.maximum(0,1-rmse)))))
    return rows


def main(args):
    out=args.out
    torch.set_num_threads(4)
    metadata=json.loads((args.cache/'metadata.json').read_text())
    checkpoint=torch.load(out/'best-model.pt',weights_only=True,map_location='cuda')
    config=checkpoint['config']
    active_panels=[p for p in metadata['panels'] if p['panel']==config['panel']]
    assert len(active_panels)==1
    model=make_model(config,len(metadata['features'])).cuda().eval()
    model.load_state_dict(checkpoint['state_dict'])
    stats=[];offset=0;development=[]
    for pm in metadata['panels']:
        if pm['panel']!=config['panel']:
            offset+=pm['nodes'];continue
        folder=args.cache/pm['panel'];old=Path(pm['old_cache'])
        x=np.load(folder/'features.npy',mmap_mode='r');y=np.load(old/'y.npy',mmap_mode='r')
        mask=np.load(old/'target.npy',mmap_mode='r')
        graph={k:torch.from_numpy(v).cuda() for k,v in np.load(out/'model-graph.npz').items()}
        graph['node_id']=torch.arange(offset,offset+pm['nodes'],device='cuda');offset+=pm['nodes']
        group={r:np.zeros((2,4),np.float64) for r in ['R1','R2','R3']}
        predictions=[]
        for d in range(196,218):
            features=torch.from_numpy(np.array(x[d:d+1],dtype=np.float32)).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                pred=model(features,graph)[0].cpu().numpy()
            base=np.asarray(x[d,:2],dtype=np.float64)*np.array([4.,2000/600])[:,None,None]
            m=mask[d];truth=y[d]
            error=(base-truth)[:,m]
            correction=(pred-base)[:,m]
            accum=group[pm['regimes'][d]]
            accum[:,0]+=m.sum()
            accum[:,1]+=(correction**2).sum(1)
            accum[:,2]+=(correction*error).sum(1)
            accum[:,3]+=(error**2).sum(1)
            predictions.append(pred)
        development.append((pm,np.stack(predictions)))
        stats.extend(group.values())
        print(json.dumps(dict(stage='calibration_development',panel=pm['panel'])),flush=True)
    stats=np.stack(stats)
    alpha=[]
    for channel in range(2):
        n,a,b,c=stats[:,channel].T
        def objective(value):
            mse=np.maximum(0,(value*value*a+2*value*b+c)/n)
            return float(np.sqrt(mse).mean())+1e-5*(value-1)**2
        fitted=minimize_scalar(objective,bounds=(0.,2.),method='bounded',options={'xatol':1e-6})
        alpha.append(float(fitted.x) if objective(fitted.x)<objective(1.) else 1.)
    alpha=np.array(alpha,np.float64)
    def evaluate_saved(entries,days,coefficients):
        rows=[]
        for pm,predictions in entries:
            x=np.load(args.cache/pm['panel']/'features.npy',mmap_mode='r')
            old=Path(pm['old_cache']);y=np.load(old/'y.npy',mmap_mode='r');mask=np.load(old/'target.npy',mmap_mode='r')
            sums={r:np.zeros(3,np.float64) for r in ['R1','R2','R3']}
            for pred,d in zip(predictions,days):
                base=np.asarray(x[d,:2],dtype=np.float64)*np.array([4.,2000/600])[:,None,None]
                value=np.maximum(0,base+coefficients[:,None,None]*(pred-base))
                delta=(value-y[d])[:,mask[d]]
                sums[pm['regimes'][d]]+=np.r_[mask[d].sum(),(delta**2).sum(1)]
            rows.extend(metric_rows(pm,sums))
        return rows
    raw_dev=evaluate_saved(development,range(196,218),np.ones(2))
    calibrated_dev=evaluate_saved(development,range(196,218),alpha)
    if score(calibrated_dev)<score(raw_dev)+1e-5:
        alpha=np.ones(2);calibrated_dev=raw_dev
    frozen=dict(coefficients=alpha.tolist(),bounds=[0,2],channels=['speed','flow_per_lane'],
                selection_dates=['2030-12-14','2031-01-04'],raw_development_score=score(raw_dev),
                calibrated_development_score=score(calibrated_dev),holdout_used=False,
                note='Only two coefficients. Chosen before reading holdout prediction/metric files.')
    (out/'residual-calibration.json').write_text(json.dumps(frozen,indent=2))
    pd.DataFrame(calibrated_dev).to_csv(out/'calibrated-development-by-regime.csv',index=False)
    # First access to saved holdout predictions happens after coefficient freeze.
    held=[]
    for pm in active_panels:
        with np.load(out/(pm['panel']+'-holdout.npz')) as saved:
            assert np.array_equal(saved['days'],np.arange(218,273))
            held.append((pm,saved['prediction']))
    raw=evaluate_saved(held,range(218,273),np.ones(2))
    calibrated=evaluate_saved(held,range(218,273),alpha)
    pd.DataFrame(calibrated).to_csv(out/'calibrated-holdout-by-regime.csv',index=False)
    reference=pd.read_csv(args.reference)
    reference=reference[reference.regime.isin(['R1','R2','R3'])]
    compared=pd.DataFrame(calibrated).merge(reference[['panel','regime','n','S_state']],
            on=['panel','regime'],suffixes=('_new','_transformer'),validate='one_to_one')
    assert (compared.n_new==compared.n_transformer).all() and len(compared)==3
    compared['delta']=compared.S_state_new-compared.S_state_transformer
    compared.to_csv(out/'calibrated-comparison-with-transformer.csv',index=False)
    result=dict(status='completed',raw_holdout_score=score(raw),calibrated_holdout_score=score(calibrated),
                original_transformer_score=float(compared.groupby('panel').S_state_transformer.mean().mean()),
                existing_isolated_fallback_global_reference=.9369103120317692,calibration=frozen,
                note='Calibration selected only on internal development; no holdout-driven choice between raw and calibrated candidates.')
    result['delta_vs_transformer']=result['calibrated_holdout_score']-result['original_transformer_score']
    (out/'calibrated-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    main(parser.parse_args())
