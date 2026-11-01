"""Directed observed-sequence ablations, fixed thresholds, frozen ancestors."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import r
import arrival_wait as wait
import arrival_ordinal_wait as ordinal
import arrival_wait_combined as retained
import arrival_unit as unit
from arrival_unit_directional import models as primary_models
from propagation_sequence import features as sequence,horizon_features

HERE=wait.HERE/'observed_sequence';HERE.mkdir(exist_ok=True)

def raw_cache(backend,pop,regime,meta,cases):
    dest=HERE/f'raw-{regime}-{pop}.npz'
    out=dict(np.load(dest)) if dest.exists() else {};groups={};names=None
    # Probe/full OOF rescue ancestors can exclude different links. Backfill
    # newly requested link keys rather than treating an existing cache as full.
    for c in cases:
        if f"{c['i']}_{c['link']}" not in out:groups.setdefault(c['i'],set()).add(c['link'])
    if not groups:return out
    for j,(i,ids) in enumerate(groups.items()):
        e=meta[i];slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));f,names=sequence(s)
        if j==0:
            changed=dict(s,y=1-s['y'],m=~s['m'],has_future_queue=not s['has_future_queue']);assert np.array_equal(f,sequence(changed)[0])
        for link in ids:out[f'{i}_{link}']=f[link]
        if (j+1)%40==0:print(json.dumps(dict(stage='observed_sequence',pop=pop,done=j+1,total=len(groups))),flush=True)
    np.savez_compressed(dest,**out);r.write(HERE/f'raw-{regime}-{pop}.json',dict(columns=names,inputs='13 history/masked-T rows only; actual graph direction, no distance assumed',target_poisoning_passed=True,windows=len({k.split('_')[0] for k in out})));return out

def training(months,backend,name):
    records=[]
    for pop in months:
        meta,z,protected,cases=wait.candidates(backend,pop,'early',name,True,retain_unknown=True);raw=raw_cache(backend,pop,'early',meta,cases)
        for c in cases:
            i,link=c['i'],c['link'];e=meta[i];temp=[]
            for h in range(5):
                yy=z[f'{i}_y'][:,link];mm=z[f'{i}_m'][:,link];targets={}
                ot=ordinal.ordinal_target(yy,mm,h)
                if ot is not None:targets['ordinal']=ot
                if mm[h]:targets['current']=int(not yy[h])
                if mm[h] and mm[h+1] and (yy[h] or yy[h+1]):targets['pair']=int(not yy[h])
                if not targets:continue
                x=unit.unit_features(c['x'][:-31],z[f'{i}_base'][:,link],protected[i][:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,True)
                f=np.r_[wait.timing_features(x,z[f'{i}_base'],z[f'{i}_first'],link,h,backend[3][e['panel']]['edges']),horizon_features(raw[f'{i}_{link}'],h)]
                temp.append(dict(f=f,targets=targets,date=c['date'],panel=e['panel'],w=c['w']))
            for v in temp:v['w']/=len(temp)
            records.extend(temp)
    return records

def fit(months,backend,name):
    records=training(months,backend,name);out={};info={}
    params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
    for target in ['ordinal','current','pair']:
        take=[v for v in records if target in v['targets']];x=np.array([v['f'] for v in take]);y=np.array([v['targets'][target] for v in take]);w=np.array([v['w'] for v in take]);w/=w.mean();dates=np.array([v['date'] for v in take]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);models=[]
        for k in [-1,0,1,2]:
            tr=np.ones(len(y),bool) if k==-1 else fold!=k
            model=lgb.train(params,lgb.Dataset(x[tr],label=y[tr],weight=w[tr]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{target}-{k}.txt'));models.append(model)
        out[target]=models;info[target]=dict(rows=len(y),classes=Counter(map(int,y)),features=x.shape[1])
    r.write(HERE/f'{name}-training.json',dict(months=months,targets=info,params=params,rounds=160,threshold=.5,class_boost=False,all_months_development=True));return out

def score(backend,meta,z,protected,cases,accepted,reject):
    by={}
    for j,c in enumerate(cases):
        if accepted[j]:by.setdefault(c['i'],[]).append(j)
    reports={}
    for variant,veto in reject.items():
        rows=[];TP=FP=FN=0;negative={}
        for i,e in enumerate(meta):
            parent=z[f'{i}_base']>=(.5 if e['condition']==0 else .25);p=parent.copy();m=z[f'{i}_m'];y=z[f'{i}_y'].astype(bool)
            if e['condition']==0:
                p|=protected[i]
                for j in by.get(i,[]):
                    if not veto[j]:p[cases[j]['h'],cases[j]['link']]=True
                assert not (parent&~p).any() and not (protected[i]&~p).any()
                if e['has_future_queue']:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
            if e['has_future_queue']:rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
            elif e['condition']==0:negative.setdefault(e['panel'],[]).append(bool((p&m).any()))
        s=backend[5](rows);s.update(TP=TP,FP=FP,FN=FN,onset_macro=float(np.mean([v for k,v in s['by_panel_condition'].items() if k.endswith('queue_onset')])),negative_false_any=float(np.mean([np.mean(v) for v in negative.values()])) if negative else None,lost_original_positive_cells=0,lost_rescue_positive_cells=0);reports[variant]=dict(scores=s,windows=rows)
    return reports

def evaluate(backend,pop,regime,name,models):
    meta,z,protected,cases=wait.candidates(backend,pop,regime,name);raw=raw_cache(backend,pop,regime,meta,cases);x=np.array([c['x'] for c in cases]);basefeatures=np.array([c['f'] for c in cases]);f=np.array([np.r_[c['f'],horizon_features(raw[f"{c['i']}_{c['link']}"],c['h'])] for c in cases]);accepted=primary_models(name)['committee_advance'][0].predict(x,num_threads=4)>=.5
    previous=retained.load(name)['pair'].predict(basefeatures,num_threads=4)>=.5;reject=dict(retained=previous);probabilities={}
    for target,bags in models.items():
        prob=np.array([m.predict(f,num_threads=4) for m in bags]);probabilities[target]=prob
        reject[target+'_single']=previous|(prob[0]>=.5);reject[target+'_consensus']=previous|(prob[1:]>=.5).all(0)
    reports=score(backend,meta,z,protected,cases,accepted,reject);catalog=[]
    for j,c in enumerate(cases):
        if not accepted[j]:continue
        e=meta[c['i']];h=c['h'];link=c['link'];yy=z[f"{c['i']}_y"][:,link];mm=z[f"{c['i']}_m"][:,link]
        catalog.append(dict(panel=e['panel'],origin=e['origin'],link_index=link,horizon_minutes=5*(h+1),truth=int(yy[h]) if mm[h] else None,sequence=raw[f"{c['i']}_{link}"].tolist(),wait_probabilities={k:v[:,j].tolist() for k,v in probabilities.items()},veto={k:bool(v[j]) for k,v in reject.items()}))
    r.write(HERE/f'cases-{regime}-{pop}.json',catalog);return reports

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--tag',default='observed_sequence')
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early','probe',probe);control=internal['retained']['scores'];eligible=[k for k,v in internal.items() if k!='retained' and v['scores']['TP']>=control['TP'] and v['scores']['FP']<control['FP'] and v['scores']['onset_macro']>=control['onset_macro'] and v['scores']['negative_false_any']<=control['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'retained';r.write(args.output/f'task2_{args.tag}_internal.json',dict(reports=internal,selected=selected,all_months_development=True));print(json.dumps(dict(pop='november',selected=selected,scores={k:{j:v['scores'][j] for j in ['onset_macro','TP','FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full','full',full);r.write(args.output/f'task2_{args.tag}_external.json',dict(reports=reports,selected=selected,all_months_development=True,official_truth_available=False,formal_R127_unchanged=True));print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
