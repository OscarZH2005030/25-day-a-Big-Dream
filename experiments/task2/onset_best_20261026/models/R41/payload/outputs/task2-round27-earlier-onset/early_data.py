"""Append earlier Onset views; preserve original 555 inputs/labels and row order."""
import json,hashlib
from pathlib import Path
import numpy as np
from task2q.cache import atomic_json
from run_round2 import sample,PANELS
from coverage_data import matrix
from special_features import latest_state

def build(p,c,entries,work,project):
    assert c==0
    folder=work/f'{p}-{c}';folder.mkdir(parents=True,exist_ok=True)
    source=project/f'work/round10-phase/{p}-{c}'
    keys=['X','y','w','early','window_positive']
    done=folder/'complete.json'
    if done.exists():
        audit=json.loads(done.read_text(encoding='utf8'))
        return {k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in keys+['w_early']},audit,audit
    old={k:np.load(source/(k+'.npy'),mmap_mode='r') for k in keys}
    meta=json.loads((source/'complete.json').read_text(encoding='utf8'))
    new=sorted([e for e in entries if e['panel']==p],key=lambda e:e['origin'])
    assert new and all(e['condition']==0 and e['has_future_queue'] and e['origin']<'2030-12' for e in new)
    assert not ({e['origin'] for e in new}&{e['origin'] for e in meta['windows']})
    counts={b:sum(r['positive']==b for r in meta['windows']) for b in [False,True]}
    total={False:counts[False],True:counts[True]+len(new)}
    earlycounts={b:sum(r['positive']==b and r['early'] for r in meta['windows']) for b in [False,True]}
    earlytotal={False:earlycounts[False],True:earlycounts[True]+sum(e['origin'][:10]<'2030-10-01' for e in new)}
    ow=np.asarray(old['w']).copy();ow[old['window_positive'].astype(bool)]*=counts[True]/total[True]
    nx=[];ny=[];nw=[];ne=[];windows=list(meta['windows']);cursor=len(old['y'])
    rng=np.random.default_rng(20261027+27000+PANELS.index(p)*2)
    for e in new:
        s=sample(e);assert s['condition']==0 and s['has_future_queue']
        xx=matrix(s);assert xx.shape[1]==555 and np.isfinite(xx).all()
        m=s['m'].ravel();y=s['y'].ravel();keep=np.flatnonzero(m&(y>0));negative=np.flatnonzero(m&~(y>0))
        last,_=latest_state(s);risk=np.tile((last<.9)|(s['x'][-2,:,1]>.7)|(s['x'][-2,:,2]>.20),6)
        a=negative[risk[negative]];b=negative[~risk[negative]];ka=min(96,len(a));kb=min(192-ka,len(b));ka=min(192-kb,len(a))
        aa=rng.choice(a,ka,replace=False) if ka else np.array([],int);bb=rng.choice(b,kb,replace=False) if kb else np.array([],int)
        ids=np.concatenate([keep,aa,bb]);inv=np.concatenate([np.ones(len(keep)),np.full(ka,len(a)/ka if ka else 0),np.full(kb,len(b)/kb if kb else 0)])
        assert len(np.unique(ids))==len(ids) and int(y[ids].sum())==int((y*m).sum())>0
        weights=(inv/m.sum()*.8/total[True]).astype('float32');assert np.isclose(weights.sum(),.8/total[True],rtol=1e-5)
        k=len(ids);early=e['origin'][:10]<'2030-10-01'
        nx.append(xx[ids]);ny.append(y[ids]);nw.append(weights);ne.extend([early]*k)
        windows.append(dict(origin=e['origin'],positive=True,source='missing_early_onset_view',start=cursor,end=cursor+k,early=early));cursor+=k
    data=dict(X=np.concatenate([old['X'],*nx]),y=np.concatenate([old['y'],*ny]),w=np.concatenate([ow,*nw]),early=np.concatenate([old['early'],np.asarray(ne,dtype='uint8')]),window_positive=np.concatenate([old['window_positive'],np.ones(sum(map(len,ny)),dtype='uint8')]))
    E=data['early'].astype(bool);P=data['window_positive'].astype(bool);ew=np.zeros_like(data['w'])
    for positive in [False,True]:
        ii=E&(P==positive);ew[ii]=data['w'][ii]*(total[positive]/earlytotal[positive])
    data['w_early']=ew
    assert np.array_equal(data['X'][:len(old['y'])],old['X']) and np.array_equal(data['y'][:len(old['y'])],old['y'])
    assert (data['w']>0).all() and (ew[E]>0).all()
    assert np.isclose(data['w'].sum(),1,atol=1e-5) and np.isclose(data['w'][P].sum(),.8,atol=1e-5)
    assert np.isclose(ew.sum(),1,atol=1e-5) and np.isclose(ew[P].sum(),.8,atol=1e-5)
    audit=dict(panel=p,condition=c,rows=len(data['y']),features=555,original_rows=len(old['y']),added_rows=sum(map(len,ny)),added_windows=len(new),added_early_windows=sum(e['origin'][:10]<'2030-10-01' for e in new),original_features_labels_order_exact=True,all_original_and_added_positives_retained=True,full_mass=float(data['w'].sum()),full_positive_mass=float(data['w'][P].sum()),early_mass=float(ew.sum()),early_positive_mass=float(ew[P].sum()),early_normalization_June_September_only=True,original_within_window_IPW_unchanged=True,source_sha256={k:hashlib.sha256((source/(k+'.npy')).read_bytes()).hexdigest() for k in keys},windows=windows)
    for k,v in data.items():np.save(folder/(k+'.npy'),v)
    atomic_json(done,audit)
    return data,audit,audit
