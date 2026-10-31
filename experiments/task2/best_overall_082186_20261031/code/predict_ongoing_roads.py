"""Portable road-specialist inference from causal274 features and frozen R127."""
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb

def probability(root,features,panels,parent,candidate):
    root=Path(root);cal=json.loads((root/'January_results.json').read_text(encoding='utf8'))['candidates'][candidate]
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'));cfg=cal['config'];x=np.asarray(features,dtype='float32');panels=np.asarray(panels)
    assert x.ndim==2 and x.shape[1]==274 and len(panels)==len(x)
    assert np.asarray(parent).shape==(len(x),6)
    result=np.empty((len(x),6),dtype='float64')
    for panel in np.unique(panels):
        ids=np.flatnonzero(panels==panel);acc=np.zeros((len(ids),6))
        for seed in protocol['seeds']:
            model=lgb.Booster(model_file=str(root/f"final_{cfg['capacity']}_{seed}"/f'{panel}.txt'))
            for h in range(6):acc[:,h]+=model.predict(np.column_stack([x[ids],np.full(len(ids),(h+1)*5)]),num_threads=8)/len(protocol['seeds'])
        result[ids]=acc
    result=result.astype('float32')
    if cfg['mix']=='odds50':
        a=np.clip(np.asarray(parent),1e-6,1-1e-6);b=np.clip(result,1e-6,1-1e-6);z=.5*(np.log(a)-np.log1p(-a)+np.log(b)-np.log1p(-b));result=np.exp(-np.logaddexp(0.,-z)).astype('float32')
    return result,float(cal['threshold'])

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--models',required=True);ap.add_argument('--features',required=True,help='npz with X[N,274], panel[N], parent[N,6]');ap.add_argument('--candidate',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    with np.load(args.features) as z:p,t=probability(args.models,z['X'],z['panel'],z['parent'],args.candidate)
    np.savez_compressed(args.output,probability=p,prediction=p>=t,threshold=t)
