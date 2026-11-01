"""Train-only empirical P(queue next30 | reliably clear now), with negatives.
Profiles for training exclude entire target date. No future held-out dates read.
"""
from datetime import date
import numpy as np
from core import ROOT,r
from episodes import load
HERE=ROOT/'work/task2-cascade/suggested-feature-cache';HERE.mkdir(exist_ok=True)

def profile(backend,months,name):
    dest=HERE/f'{name}-hazard.npz'
    if dest.exists():return dict(np.load(dest))
    roster=sorted({(e['panel'],e['origin'][:10]) for pop in months for e in load(pop,'early')[0]});out={}
    for panel,d in roster:
        v=backend[1].load_day(panel,d);net=backend[3][panel];good=v['eligible']&np.isfinite(v['speed'])&np.isfinite(v['quality'])&(v['quality']>=75);clear=good&(v['speed']>net['threshold']);queued=good&(v['speed']<=net['threshold']);events=np.zeros((len(net['link_ids']),6,2));counts=np.zeros_like(events);kind=int(date.fromisoformat(d).weekday()>=5)
        for t in range(12,282):
            # Reliable negatives must have a completely observed next30.
            eligible=clear[t]&good[t+1:t+7].all(0);event=queued[t+1:t+7].any(0)&eligible;band=t//48;counts[:,band,kind]+=eligible;events[:,band,kind]+=event
        out[panel+'|'+d+'|n']=counts;out[panel+'|'+d+'|e']=events
    np.savez_compressed(dest,**out);r.write(HERE/f'{name}-hazard.json',dict(months=months,roster=roster,conditional='reliably clear now; next30 all score-eligible quality>=75',negative_days_included=True,training_features_strictly_earlier_dates=True,scope='all time slots on dates represented by training roster; date selection is not population-independent'))
    return out

def hazard_features(store,panel,d,slot,exclude_date=False):
    kind=int(date.fromisoformat(d).weekday()>=5);band=slot//48;nn=[];ee=[]
    for key,n in store.items():
        p,dd,what=key.split('|')
        if p!=panel or what!='n' or (exclude_date and dd>=d):continue
        nn.append(n);ee.append(store[p+'|'+dd+'|e'])
    if not nn:
        nlinks=next(v.shape[0] for k,v in store.items() if k.startswith(panel+'|'))
        return np.tile([.5,0.,0.,.5,0.,0.],(nlinks,1)).astype('float32')
    n=np.sum(nn,axis=0);e=np.sum(ee,axis=0);a=n[:,band,kind];b=e[:,band,kind];alln=n.sum((1,2));alle=e.sum((1,2))
    return np.column_stack([(b+.5)/(a+1),np.log1p(a),a>0,(alle+.5)/(alln+1),np.log1p(alln),alln>0]).astype('float32')
