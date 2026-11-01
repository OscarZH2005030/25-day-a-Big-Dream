"""Shadow-only rescue guard with explicit parent/generator disagreement.
All inputs are frozen predictions or observed history; no evaluation labels.
"""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
import shadow_rescue_precision as base
import arrival_context
from core import ROOT,r

ORIGINAL=base.risk_features
HERE=base.HERE/'trajectory_context';HERE.mkdir(exist_ok=True)
LOOKUP={}

def context_features(b,q,event,first,before,after,conditional,joint,rescue,h):
    b,q,first,before,after,conditional,joint=[np.asarray(v,float) for v in (b,q,first,before,after,conditional,joint)]
    assert all(v.shape==(6,) for v in (b,q,first,before,after,conditional,joint))
    event=np.asarray(event,float);assert event.shape==(7,) and 0<=h<6
    E=1-event[0];seed=np.flatnonzero(b>=.5)
    kind=0 if not len(seed) else 1 if h<seed[0] else 2 if h>seed[-1] else 3
    trajectory=np.r_[b,q,event,first,before,after,conditional,joint,q*E-b,joint-q*E]
    ids=np.flatnonzero(rescue);assert h in ids
    start=end=h
    while start>0 and rescue[start-1]:start-=1
    while end<5 and rescue[end+1]:end+=1
    local=[b[h],b[max(0,h-1)],b[min(5,h+1)],b.max(),b.mean(),E,E-b.max(),q[h]*E,joint[h],joint[h]-b[h],q[h]*E-joint[h],first[:h+1].sum()*E,before[h]*E+1-E,after[h]*E,(h-seed[0])/5 if len(seed) else -1,(h-seed[-1])/5 if len(seed) else -1,h/5,float(h==5)]
    extent=[(end-start+1)/6,(end-start)/5,start/5,end/5,float(end==5),float(start==0),(h-start)/max(end-start,1)]
    semantic=np.r_[np.eye(4)[kind],np.asarray(rescue,float),local,extent]
    assert trajectory.shape==(61,) and semantic.shape==(35,)
    assert np.isfinite(np.r_[trajectory,semantic]).all()
    return trajectory.astype('float32'),semantic.astype('float32')

def enriched(legacy,margin,raw,h,view):
    t,s=LOOKUP[np.asarray(legacy).tobytes()]
    old=ORIGINAL(legacy,margin,raw,h,view)
    return np.r_[old,t,s] if view=='observed' else np.r_[old,t]

def prepare(pop,regime,name,training=False):
    meta,old=base.load(pop,regime);pm,z,_=arrival_context.cache(None,pop,regime)
    assert [e['origin'] for e in meta]==[e['origin'] for e in pm]
    if training:accepted,fold,models=base.oof_models(name)
    else:model=lgb.Booster(model_file=str(ROOT/'work/task2-cascade'/f'margin-{name}.txt'))
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        x,ri,hi=old[f'{i}_Z'],old[f'{i}_ri'],old[f'{i}_hi']
        if not len(x):continue
        if training:
            a=accepted[f'{pop}_{i}'];score=models[fold[e['origin'][:10]]].predict(x,num_threads=4)-x[:,3];assert np.array_equal(a,score<=0)
        else:a=model.predict(x,num_threads=4)-x[:,3]<=0
        mask=np.zeros_like(old[f'{i}_base'],bool);mask[hi[a],ri[a]]=True
        for j in np.flatnonzero(a):
            link,h=int(ri[j]),int(hi[j])
            v=context_features(old[f'{i}_base'][:,link],old[f'{i}_q'][:,link],old[f'{i}_event'][link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],mask[:,link],h)
            key=x[j].tobytes()
            if key in LOOKUP:assert all(np.array_equal(a,b) for a,b in zip(LOOKUP[key],v))
            LOOKUP[key]=v

def fit(months,backend,name):
    LOOKUP.clear()
    for pop in months:prepare(pop,'early',name,True)
    rows=base.training(months,backend,name);dates=np.array([c['date'] for c in rows]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);w=np.array([c['w'] for c in rows]);w/=w.mean();y=np.array([c['not_queued'] for c in rows]);out={}
    for view in ['legacy','observed']:
        x=np.array([c['views'][view] for c in rows]);bags=[]
        for k in [-1,0,1,2]:
            tr=np.ones(len(y),bool) if k==-1 else fold!=k
            params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
            m=lgb.train(params,lgb.Dataset(x[tr],label=y[tr],weight=w[tr]),num_boost_round=160);m.save_model(str(HERE/f'{name}-{view}-{k}.txt'));bags.append(m)
        out[view+'_binary']=bags
    r.write(HERE/f'{name}-training.json',dict(months=months,cells=len(rows),TP=int((1-y).sum()),FP=int(y.sum()),widths={v:len(rows[0]['views'][v]) for v in ['legacy','observed']},OOF=True,threshold=.5,class_weight=False,shadow_only=True,ancestors_not_nested=True))
    return out

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();protocol=r.read(args.output/'task2_origin_stress_frozen.json');base.check_models(protocol);backend=r.setup(args);base.HERE=HERE;base.risk_features=enriched
    probe=fit(['october'],backend,'probe');LOOKUP.clear();prepare('november','early','probe');internal=base.evaluate(backend,'november','early','probe',probe)
    b=internal['retained']['scores'];eligible=[k for k,v in internal.items() if k!='retained' and v['scores']['lost_retained_TP']==0 and v['scores']['all_onset_FP']<b['all_onset_FP'] and v['scores']['onset_macro']>=b['onset_macro'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'retained'
    r.write(args.output/'task2_shadow_trajectory_internal.json',dict(reports=internal,selected=selected,shadow_only=True))
    print(json.dumps(dict(pop='november',selected=selected,scores={k:{j:v['scores'][j] for j in ['TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        LOOKUP.clear();prepare(pop,'full','full');reports[pop]=base.evaluate(backend,pop,'full','full',full)
        r.write(args.output/'task2_shadow_trajectory_external.json',dict(reports=reports,selected=selected,shadow_only=True,official_truth_available=False,repeated_development_months=True,default_unchanged=True))
        print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    base.check_models(protocol)
if __name__=='__main__':main()
