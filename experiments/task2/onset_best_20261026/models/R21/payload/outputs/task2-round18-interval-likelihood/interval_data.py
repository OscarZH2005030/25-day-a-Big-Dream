import json
from pathlib import Path
import numpy as np
from coverage_data import matrix
from run_round2 import sample,PANELS
from task2q.cache import atomic_json
from interval_core import encode,ALL
SEED=20261027
def features(s):
    n=s['x'].shape[1];v=matrix(s)[:n];road=np.zeros((n,8),dtype='float32');road[:,PANELS.index(s['panel'])]=1
    return np.concatenate([v[:,:185],v[:,191:],road],axis=1)
def build(entries,work,progress):
    overall=work/'complete.json'
    if overall.exists():return {k:np.load(work/(k+'.npy'),mmap_mode='r') for k in ['X','code','w','early','positive']},json.loads(overall.read_text(encoding='utf8'))
    audits=[];windows=[];cursor=0
    for p in PANELS:
        folder=work/p;folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json'
        if done.exists():audit=json.loads(done.read_text(encoding='utf8'))
        else:
            es=[e for e in entries if e['panel']==p and e['condition']==0];counts={b:sum(e['has_future_queue']==b for e in es) for b in [False,True]};rng=np.random.default_rng(SEED+4100+PANELS.index(p));chunks={k:[] for k in ['X','code','w','early','positive']};local_windows=[];local_rows=0;complete_positive=0;compatible_complete_positive=0;discarded_positive=0;kept_positive=0;ambiguous_positive=0;unconstrained=0
            for i,e in enumerate(es):
                s=sample(e);code=encode(s['y'].astype(bool),s['m']);positive=s['y'].astype(bool).any(0);informative=(code>0)&(code<ALL);exact=(code&(code-1))==0;pos=np.flatnonzero(positive&informative);neg=np.flatnonzero(~positive&informative);k=min(192,len(neg));ids=np.concatenate([pos,rng.choice(neg,k,replace=False) if k else np.array([],int)]);inv=np.concatenate([np.ones(len(pos)),np.full(k,len(neg)/k if k else 0)]);mass=(.8 if e['has_future_queue'] else .2)/counts[e['has_future_queue']];w=inv/max(int(informative.sum()),1)*mass;xx=features(s);assert xx.shape[1]==557;chunks['X'].append(xx[ids]);chunks['code'].append(code[ids]);chunks['w'].append(w.astype('float32'));chunks['early'].append(np.full(len(ids),e['origin'][:10]<'2030-10-01',dtype='uint8'));chunks['positive'].append(np.full(len(ids),e['has_future_queue'],dtype='uint8'));local_windows.append(dict(origin=e['origin'],positive=e['has_future_queue'],early=e['origin'][:10]<'2030-10-01',start=local_rows,end=local_rows+len(ids)));local_rows+=len(ids);cp=positive&s['m'].all(0);complete_positive+=int(cp.sum());compatible_complete_positive+=int((cp&(code>0)).sum());discarded_positive+=int((positive&(code==0)).sum());kept_positive+=len(pos);ambiguous_positive+=int((positive&informative&~exact).sum());unconstrained+=int((code==ALL).sum())
                if (i+1)%128==0:progress('building_interval_rows',panel=p,windows=i+1,total=len(es))
            arrays={k:np.concatenate(v) for k,v in chunks.items()};assert np.isfinite(arrays['X']).all();assert np.array_equal(arrays['code'],arrays['code'].astype('float32').astype('uint32'))
            for k,v in arrays.items():np.save(folder/(k+'.npy'),v)
            audit=dict(panel=p,rows=local_rows,windows=local_windows,complete_positive_links=complete_positive,compatible_complete_positive=compatible_complete_positive,compatible_positive_fraction=compatible_complete_positive/max(complete_positive,1),positive_links_retained=kept_positive,ambiguous_positive_links_retained=ambiguous_positive,incompatible_positive_links_excluded_from_new_head=discarded_positive,unconstrained_links_excluded=unconstrained,counts={str(b):n for b,n in counts.items()});atomic_json(done,audit);del arrays,chunks
        audits.append(audit)
        for r in audit['windows']:windows.append(dict(r,panel=p,start=r['start']+cursor,end=r['end']+cursor))
        cursor+=audit['rows']
    maps={k:np.lib.format.open_memmap(work/(k+'.npy'),mode='w+',dtype=dtype,shape=(cursor,557) if k=='X' else (cursor,)) for k,dtype in [('X','float32'),('code','uint32'),('w','float32'),('early','uint8'),('positive','uint8')]};cursor=0
    for a in audits:
        end=cursor+a['rows']
        for k,dest in maps.items():dest[cursor:end]=np.load(work/a['panel']/(k+'.npy'),mmap_mode='r')
        cursor=end
    for dest in maps.values():dest.flush()
    meta=dict(rows=cursor,features=557,patterns=22,panels=[{k:v for k,v in a.items() if k!='windows'} for a in audits],windows=windows,control='same selected supervision rows restricted to singleton compatible interval labels',partial='all informative compatible interval label sets; optimize negative log summed probability',negative_sampling='<=192 informative negative links/window with inverse-inclusion weights; all compatible observed-positive links retained',scope='6—11month expanded onset origins only; noisy proxy, overlapping windows correlated; original all-positive baseline unchanged');atomic_json(overall,meta);return {k:np.load(work/(k+'.npy'),mmap_mode='r') for k in maps},meta
