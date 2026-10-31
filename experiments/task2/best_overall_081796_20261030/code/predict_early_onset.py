"""Portable inference on996 causal features; no experiment data dependency."""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from onset_interval_core import softmax,interval_distribution,PATTERNS,interval_decode
SEEDS=[20261030,20261031,20261101]
def timing(model,x):
    parts=[]
    for a in range(0,len(x),16384):
        z=np.asarray(x[a:a+16384]);start=softmax(model.predict(np.column_stack([z,np.zeros(len(z))]),raw_score=True,num_threads=8));end=softmax(model.predict(np.column_stack([z,np.ones(len(z))]),raw_score=True,num_threads=8));parts.append(interval_distribution(start,end)@PATTERNS)
    return np.concatenate(parts)
def probability(root,x,name):
    root=Path(root);r=json.loads((root/'January_results.json').read_text())['candidates'][name];cfg=r['config'];ps=[];assert x.shape[1]==996
    for seed in SEEDS:
        em=lgb.Booster(model_file=str(root/f'shared/final_{seed}_event.txt'));z=np.concatenate([em.predict(x[a:a+16384],raw_score=True,num_threads=8) for a in range(0,len(x),16384)]);e=np.exp(-np.logaddexp(0.,-z))
        if cfg['head'] in ['control','early']:
            tm=lgb.Booster(model_file=str(root/f'shared/final_{seed}_timing.txt'));q=timing(tm,x[:,:940])
            if cfg['head']=='early':
                model=lgb.Booster(model_file=str(root/f'final_early_{seed}/model.txt'));ss=[]
                for h in range(3):ss.append(np.concatenate([model.predict(np.column_stack([x[a:a+16384],np.full(len(x[a:a+16384]),h)]),num_threads=8) for a in range(0,len(x),16384)]))
                q[:,:3]=(1-cfg['blend'])*q[:,:3]+cfg['blend']*np.stack(ss,1)
        else:
            model=lgb.Booster(model_file=str(root/f"final_{cfg['head']}_{seed}/model.txt"));q=timing(model,x)
        ps.append((e[:,None]*q).astype('float32'))
    return np.mean(ps,0),np.array(r['selection']['threshold'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--candidate',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();p,t=probability(a.models,np.load(a.features,mmap_mode='r'),a.candidate);a.output.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(a.output,probability=p,queue_pred=interval_decode(p,t).astype('uint8'),threshold=t)
