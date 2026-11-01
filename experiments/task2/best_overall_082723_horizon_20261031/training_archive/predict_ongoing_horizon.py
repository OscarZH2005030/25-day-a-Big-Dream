"""Portable composition of frozen900-tree parents and horizon residual logits."""
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb

def probability(root,features,panels,parent,candidate):
    root=Path(root);cal=json.loads((root/'January_results.json').read_text())['candidates'][candidate];protocol=json.loads((root/'protocol.json').read_text());cfg=cal['config'];x=np.asarray(features,dtype='float32');panels=np.asarray(panels);assert x.shape==(len(panels),274) and np.asarray(parent).shape==(len(x),6);result=np.empty((len(x),6),dtype='float32')
    for panel in np.unique(panels):
        ids=np.flatnonzero(panels==panel);acc=np.zeros((len(ids),6))
        for seed in protocol['seeds']:
            base=lgb.Booster(model_file=str(root/'base_models'/f'final_{seed}'/f'{panel}.txt'))
            for h in range(6):
                xx=np.column_stack([x[ids],np.full(len(ids),(h+1)*5)]);res=lgb.Booster(model_file=str(root/f"final_{cfg['capacity']}_{seed}"/f'{panel}-h{(h+1)*5}.txt'));z=base.predict(xx,raw_score=True,num_threads=8)+cfg['alpha']*res.predict(xx,raw_score=True,num_threads=8);acc[:,h]+=np.exp(-np.logaddexp(0.,-z))/len(protocol['seeds'])
        result[ids]=acc
    a=np.clip(parent,1e-6,1-1e-6);b=np.clip(result,1e-6,1-1e-6);z=.5*(np.log(a)-np.log1p(-a)+np.log(b)-np.log1p(-b));return np.exp(-np.logaddexp(0.,-z)).astype('float32'),float(cal['threshold'])

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--models',required=True);ap.add_argument('--features',required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    with np.load(a.features) as z:p,t=probability(a.models,z['X'],z['panel'],z['parent'],a.candidate)
    np.savez_compressed(a.output,probability=p,prediction=p>=t,threshold=t)
