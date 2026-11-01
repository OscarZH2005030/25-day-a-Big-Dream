"""Predict a one-slot deferral instead of demanding strong onset evidence.

Only veto new frozen-committee advances. Parent and margin positives are hard
preserved. Future measurements appear only in supervised targets/diagnostics.
"""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import r
import arrival_unit as unit
import arrival_context as context
import advance_context_reliability as traffic
from arrival_safe import oof_rescues
from arrival_unit_directional import models as load_unit
from episodes import load
from margin import Margin
from onset_phase import BASE

HERE=context.HERE/'one_slot_wait';HERE.mkdir(exist_ok=True)

def timing_features(x,b,first,link,h,edges):
    """Measured causal context + relative forecast timing across actual edges."""
    assert 0<=h<5
    prev=max(h-1,0);nxt=h+1;f=first[link];cdf=np.cumsum(f)
    hazard=f/np.maximum(1-np.r_[0,cdf[:-1]],1e-6)
    local=np.r_[f[prev],f[h],f[nxt],cdf[h],cdf[nxt],hazard[h],hazard[nxt],f[nxt]-f[h],b[h,link],b[nxt,link],b[nxt,link]-b[h,link],h/5]
    src,dst=edges;parts=[]
    for ids in [src[dst==link],dst[src==link]]:
        if not len(ids):parts.extend([0.]*10+[1.]);continue
        p=b[:,ids];mf=first[ids];parts.extend([p[h].mean(),p[nxt].mean(),(p[nxt]-p[h]).mean(),(p[h]>=.5).mean(),(p[nxt]>=.5).mean(),mf[:,h].mean(),mf[:,nxt].mean(),(mf[:,nxt]-mf[:,h]).mean(),(mf.argmax(1)<h).mean(),(mf.argmax(1)==nxt).mean(),0.])
    return np.r_[traffic.aligned_features(x[None])[0],local,parts].astype('float32')

def target(y,m,h):
    if not m[h]:return None
    if y[h]:return 0 # already queued at proposed cell, including weak precursors
    if not m[h+1]:return None # never turn missing next truth into an absence
    return 1 if y[h+1] else 2

def candidates(backend,pop,regime,name,training=False,retain_unknown=False):
    meta,z,c=context.cache(backend,pop,regime);_,old=load(pop,regime)
    rescues=oof_rescues([pop],backend,name) if training else None
    margin=None if training else Margin(lgb.Booster(model_file=str(BASE/f'margin-{name}.txt')))
    panelcounts=Counter(e['panel'] for e in meta);records=[];protected={}
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        ri,hi,xx=old[f'{i}_ri'],old[f'{i}_hi'],old[f'{i}_Z']
        a=rescues[f'{pop}_{i}'] if training else (margin.predict(xx,num_threads=4)>=.5 if len(xx) else np.zeros(0,bool))
        rescue=np.zeros_like(z[f'{i}_base'],bool);rescue[hi[a],ri[a]]=True;protected[i]=rescue
        pending=[]
        for row,link in enumerate(c[f'{i}_ri']):
            if c[f'{i}_direction'][row]!=0:continue
            proposal=unit.unit_proposal(z[f'{i}_base'][:,link],c[f'{i}_pred'][row],0)
            if proposal is None:continue
            h,_=proposal
            if rescue[h,link]:continue
            y=target(z[f'{i}_y'][:,link],z[f'{i}_m'][:,link],h)
            if training and y is None and not retain_unknown:continue
            x=unit.unit_features(c[f'{i}_Z'][row],z[f'{i}_base'][:,link],rescue[:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,True)
            f=timing_features(x,z[f'{i}_base'],z[f'{i}_first'],link,h,backend[3][e['panel']]['edges'])
            pending.append(dict(i=i,link=int(link),h=h,x=x,f=f,y=y,date=e['origin'][:10]))
        for v in pending:v['w']=1/panelcounts[e['panel']]/len(pending)
        records.extend(pending)
    return meta,z,protected,records

def fit(months,backend,name):
    records=[]
    for pop in months:records+=candidates(backend,pop,'early',name,True)[3]
    x=np.array([v['f'] for v in records]);y=np.array([v['y'] for v in records]);w=np.array([v['w'] for v in records]);w/=w.mean();dates=np.array([v['date'] for v in records]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates])
    params=dict(num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
    multi=lgb.train(dict(params,objective='multiclass',num_class=3),lgb.Dataset(x,label=y,weight=w),num_boost_round=160);multi.save_model(str(HERE/f'{name}-multi.txt'))
    pairs=y<2;models=[]
    for k in [-1,0,1,2]:
        take=pairs if k==-1 else pairs&(fold!=k)
        model=lgb.train(dict(params,objective='binary'),lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-pair-{k}.txt'));models.append(model)
    r.write(HERE/f'{name}-training.json',dict(months=months,rows=len(y),features=x.shape[1],class_counts=Counter(map(int,y)),classes=['queued_now','not_queued_now_queued_next','not_queued_now_or_next'],pair_rows=int(pairs.sum()),thresholds=dict(onset=.5,ongoing=.25,primary=.5,wait_veto=.5),rounds=160,params=params,unknown_next_negatives_excluded=True,no_class_boost=True,negative_veto_only=True))
    return dict(multi=multi,pair=models[0],pair_consensus=models[1:])

def veto(models,x):
    return dict(multi=models['multi'].predict(x,num_threads=4)[:,1]>=.5,pair=models['pair'].predict(x,num_threads=4)>=.5,pair_consensus=(np.array([m.predict(x,num_threads=4) for m in models['pair_consensus']])>=.5).all(0))

def evaluate(backend,pop,regime,name,models):
    meta,z,protected,cases=candidates(backend,pop,regime,name)
    primary=load_unit(name)['committee_advance'][0];x=np.array([v['x'] for v in cases]);f=np.array([v['f'] for v in cases]);accepted=primary.predict(x,num_threads=4)>=.5;reject=veto(models,f)
    bywindow={};catalog=[]
    for j,c in enumerate(cases):
        if not accepted[j]:continue
        bywindow.setdefault(c['i'],[]).append(j);e=meta[c['i']]
        catalog.append(dict(panel=e['panel'],origin=e['origin'],link_index=c['link'],horizon_minutes=5*(c['h']+1),target=c['y'],timing_features=c['f'][-34:].tolist(),veto={k:bool(v[j]) for k,v in reject.items()}))
    reports={}
    for variant in ['base','margin','committee_advance']+list(reject):
        rows=[];TP=FP=FN=lost=lost_rescue=0;negative={}
        for i,e in enumerate(meta):
            y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m'];parent=z[f'{i}_base']>=(.5 if e['condition']==0 else .25);p=parent.copy()
            if e['condition']==0:
                if variant!='base':p|=protected[i]
                if variant not in ['base','margin']:
                    for j in bywindow.get(i,[]):
                        if variant=='committee_advance' or not reject[variant][j]:p[cases[j]['h'],cases[j]['link']]=True
                lost+=int((parent&~p).sum())
                if variant!='base':lost_rescue+=int((protected[i]&~p).sum())
                if e['has_future_queue']:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
            if e['has_future_queue']:rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
            elif e['condition']==0:negative.setdefault(e['panel'],[]).append(bool((p&m).any()))
        s=backend[5](rows);s.update(onset_macro=float(np.mean([v for k,v in s['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,lost_original_positive_cells=lost,lost_rescue_positive_cells=lost_rescue,negative_false_any=float(np.mean([np.mean(v) for v in negative.values()])) if negative else None)
        assert lost==0 and lost_rescue==0
        reports[variant]=dict(scores=s,windows=rows)
    r.write(HERE/f'cases-{regime}-{pop}.json',catalog)
    return reports

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early','probe',probe);control=internal['committee_advance']['scores']
    eligible=[k for k in probe if internal[k]['scores']['TP']>=control['TP'] and internal[k]['scores']['FP']<control['FP'] and internal[k]['scores']['onset_macro']>=control['onset_macro']]
    selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'committee_advance'
    r.write(args.output/'task2_one_slot_wait_internal.json',dict(reports=internal,selected=selected,all_months_development=True))
    print(json.dumps(dict(pop='november',selected=selected,scores={k:{j:v['scores'][j] for j in ['onset_macro','TP','FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full','full',full)
        r.write(args.output/'task2_one_slot_wait_external.json',dict(reports=reports,selected=selected,all_months_development=True,official_truth_available=False,formal_R127_unchanged=True))
        print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
