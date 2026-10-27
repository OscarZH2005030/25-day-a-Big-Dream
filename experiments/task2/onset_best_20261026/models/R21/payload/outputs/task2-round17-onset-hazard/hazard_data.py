"""First-arrival risk labels with explicit censoring; all features remain <=T."""
import json,os
from pathlib import Path
import numpy as np
from coverage_data import matrix
from run_round2 import sample,PANELS
from task2q.cache import atomic_json
SEED=20261027
def risk_mask(mask,queue):
    risk=np.empty_like(mask);prefix=np.ones(mask.shape[1],bool)
    for h in range(6):risk[h]=prefix&mask[h];prefix&=mask[h]&~queue[h]
    return risk
def probability(model,x):
    hazard=model.predict(x,num_threads=4).astype('float32').reshape(6,-1)
    return (1-np.cumprod(1-hazard,axis=0)).astype('float32')
def group_data(entries,p,work,progress):
    folder=work/p;folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json'
    if done.exists():
        meta=json.loads(done.read_text(encoding='utf8'));return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['X','y','w','early','window_positive']},meta
    selected=[e for e in entries if e['panel']==p and e['condition']==0];counts={b:sum(e['has_future_queue']==b for e in selected) for b in [False,True]};rng=np.random.default_rng(SEED+3100+PANELS.index(p));chunks={k:[] for k in ['X','y','w','early','window_positive']};windows=[];cursor=0;positives=0;original_positive=0;ambiguous_positive=0;complete_positive=0;persistent_positive=0
    for i,e in enumerate(selected):
        s=sample(e);m=s['m'];y=s['y'].astype(bool);risk=risk_mask(m,y)
        # A missing/low-quality earlier target censors the first-arrival label.
        # It is never interpreted as an observed free-flow frame.
        positive=np.flatnonzero(risk.ravel()&y.ravel());negative=np.flatnonzero(risk.ravel()&~y.ravel());k=min(192,len(negative));neg=rng.choice(negative,k,replace=False) if k else np.array([],int);ids=np.concatenate([positive,neg]);inverse=np.concatenate([np.ones(len(positive)),np.full(k,len(negative)/k if k else 0)]);pos=e['has_future_queue'];weight=inverse/max(int(risk.sum()),1)*(.8 if pos else .2)/counts[pos];xx=matrix(s)
        if len(ids):
            chunks['X'].append(xx[ids]);chunks['y'].append(y.ravel()[ids].astype('float32'));chunks['w'].append(weight.astype('float32'));chunks['early'].append(np.full(len(ids),e['origin'][:10]<'2030-10-01',dtype='uint8'));chunks['window_positive'].append(np.full(len(ids),pos,dtype='uint8'))
        windows.append(dict(origin=e['origin'],source=e['source'],positive=pos,early=e['origin'][:10]<'2030-10-01',start=cursor,end=cursor+len(ids)));cursor+=len(ids);positives+=len(positive);original_positive+=int((m&y).sum());ambiguous_positive+=int(((m&y).any(0)&~(risk&y).any(0)).sum());complete=m.all(0)&y.any(0);complete_positive+=int(complete.sum());persistent_positive+=int((complete&y[-1]).sum())
        if (i+1)%128==0:progress('building_hazard_rows',panel=p,windows=i+1,total=len(selected))
    data={k:np.concatenate(v) for k,v in chunks.items()};assert np.isfinite(data['X']).all();meta=dict(panel=p,condition=0,rows=cursor,features=555,windows=windows,counts={str(b):n for b,n in counts.items()},known_first_arrival_positive_rows=positives,original_queue_positive_cells=original_positive,positive_links_first_time_censored=ambiguous_positive,complete_positive_links=complete_positive,complete_positive_still_queued_at30=persistent_positive,sampling='all known first arrivals + inverse-inclusion 192 risk-negative cells per window; weights .8/.2 window-stratum mass; no future fields in features',limitation='risk set terminates at first missing/low-quality or observed queue; selection may depend on noisy measurement availability; baseline all-positive supervision stays intact')
    for k,v in data.items():np.save(folder/(k+'.npy'),v)
    atomic_json(done,meta);return data,meta
