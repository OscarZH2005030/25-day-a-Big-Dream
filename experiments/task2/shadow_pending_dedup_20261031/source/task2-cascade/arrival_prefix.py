"""Learn first-arrival edits separately; never reforecast the post-arrival tail."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import lightgbm as lgb
from core import ROOT,r
from onset_phase import HERE,BASE
from onset_timing_evaluate import link_delta
from episodes import load as old_load
from margin import Margin
from relative_support import relative_features

def propose(b,first,conditional,use):
    """MAP arrival time; delay only removes a prefix ending at an old positive.

    No parent event can disappear. No existing post-arrival tail is modified.
    """
    ids=[];directions=[];preds=[];starts=[];oldstarts=[];ranks=[];oldbits=b>=.5;rel,_=relative_features(b,positive_threshold=.5)
    for link in np.flatnonzero(use&oldbits.any(0)):
        oldfirst=np.flatnonzero(oldbits[:,link])[0];newfirst=int(np.argmax(first[link]));p=oldbits[:,link].copy()
        if newfirst>oldfirst:
            remaining=np.flatnonzero(oldbits[:,link]&(np.arange(6)>=newfirst))
            if not len(remaining):continue
            newfirst=int(remaining[0]);p[:newfirst]=False;direction=1
        elif newfirst<oldfirst:
            p[newfirst:oldfirst]|=conditional[newfirst:oldfirst,link]>=.5;p[newfirst]=True;direction=0
        else:continue
        if np.array_equal(p,oldbits[:,link]):continue
        ids.append(link);directions.append(direction);preds.append(p);starts.append(newfirst);oldstarts.append(oldfirst);changed=p!=oldbits[:,link];ranks.append(np.r_[rel[link,changed].mean(0),rel[link,changed].min(0),rel[link,changed].max(0)])
    return np.array(ids,int),np.array(directions,int),np.array(preds,bool).reshape(-1,6),np.array(starts,int),np.array(oldstarts,int),np.array(ranks,dtype='float32').reshape(-1,24)

def cache(pop,regime):
    dest=HERE/f'prefix-v2-{regime}-{pop}.npz'
    meta=r.read(HERE/f'{regime}-{pop}.json');z=dict(np.load(HERE/f'{regime}-{pop}.npz'))
    if dest.exists():return meta,z,dict(np.load(dest))
    c={}
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        b=z[f'{i}_base'];ids,direction,p,newfirst,oldfirst,rank=propose(b,z[f'{i}_first'],z[f'{i}_conditional'],z[f'{i}_unseen']);oldids=z[f'{i}_ri'];oldx=z[f'{i}_Z'];mapping={int(k):j for j,k in enumerate(oldids)};x=[];kept=[]
        for row,link in enumerate(ids):
            if int(link) not in mapping:continue
            # All proposals normally also change the full shape. If a MAP
            # prefix changes while the full-shape binary forecast does not,
            # keep the parent rather than fabricate missing summary inputs.
            v=np.r_[oldx[mapping[int(link)]],np.eye(6)[newfirst[row]],np.eye(6)[oldfirst[row]],p[row],rank[row],direction[row]];x.append(v);kept.append(row)
        kept=np.array(kept,int);width=oldx.shape[1]+43
        for k,v in dict(ri=ids[kept],direction=direction[kept],pred=p[kept],Z=np.array(x,dtype='float32').reshape(-1,width)).items():c[f'{i}_{k}']=v
    np.savez_compressed(dest,**c);return meta,z,c

def fit(months,name):
    xs=[];ys=[];ws=[];ds=[]
    for pop in months:
        meta,z,c=cache(pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ids=c[f'{i}_ri'];valid=[];labels=[]
            for row,link in enumerate(ids):
                new=z[f'{i}_base'].copy();new[:,link]=c[f'{i}_pred'][row].astype(float);delta=link_delta(z[f'{i}_y'],z[f'{i}_m'],z[f'{i}_base'],new,link)
                if delta is not None:valid.append(row);labels.append(delta>0)
            if valid:xs.append(c[f'{i}_Z'][valid]);ys.extend(labels);ds.extend(c[f'{i}_direction'][valid]);ws.extend([1/counts[e['panel']]/len(valid)]*len(valid))
    x=np.concatenate(xs);y=np.array(ys);direction=np.array(ds);w=np.array(ws);w/=w.mean();models={};info=[]
    for d in range(2):
        take=direction==d
        if take.sum()<60 or len(np.unique(y[take]))<2:continue
        params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{d}.txt'));models[d]=model;info.append(dict(direction=['advance','delay'][d],rows=int(take.sum()),beneficial=int(y[take].sum())))
    r.write(HERE/f'{name}.json',dict(months=months,heads=info,threshold=.5,class_weight_boost=False,event_existence_preserved=True,post_arrival_tail_changed=False));return models

def evaluate(meta,z,c,old,backend,models,margin,variant):
    rows=[];neg=defaultdict(list);TP=FP=FN=0;events_lost=0;actions=0
    for i,e in enumerate(meta):
        b=z[f'{i}_base'];p=b>=(.5 if e['condition']==0 else .25);y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
        if e['condition']==0:
            before=p.copy();ids=c[f'{i}_ri'];d=c[f'{i}_direction'];accept=np.zeros(len(ids),bool)
            if variant not in ['base','margin']:
                for direction,model in models.items():
                    take=d==direction
                    if variant in ['delay_only','delay_margin']:take&=d==1
                    if variant in ['advance_only','advance_margin']:take&=d==0
                    if take.any():accept[take]=model.predict(c[f'{i}_Z'][take],num_threads=4)>=.5
                p[:,ids[accept]]=c[f'{i}_pred'][accept].T;actions+=int(accept.sum())
            if variant in ['margin','prefix_margin','delay_margin','advance_margin']:
                ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];x=old[f'{i}_Z'];take=margin.predict(x,num_threads=4)>=.5 if len(x) else np.zeros(0,bool);p[hi[take],ri[take]]=True
            events_lost+=int((before.any(0)&~p.any(0)).sum())
        if not e['has_future_queue']:
            if e['condition']==0:neg[e['panel']].append(bool((p&m).any()))
            continue
        rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
        if e['condition']==0:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
    score=backend[5](rows);score.update(onset_macro=float(np.mean([v for k,v in score['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None,parent_events_deleted=events_lost,retimed_links=actions);return score,rows

def population(backend,pop,regime,models):
    meta,z,c=cache(pop,regime);_,old=old_load(pop,regime);margin=Margin(lgb.Booster(model_file=str(BASE/('margin-probe.txt' if regime=='early' else 'margin-full.txt'))));results={}
    for k in ['base','margin','advance_only','delay_only','prefix','prefix_margin','delay_margin','advance_margin']:
        scores,rows=evaluate(meta,z,c,old,backend,models,margin,k);results[k]=dict(scores=scores,windows=rows)
    return results

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);models=fit(['october'],'prefix-probe');internal=population(backend,'november','early',models);b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k!='base' and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_onset_prefix_internal.json',dict(reports=internal,selected=selected,later_month_selection=False,threshold=.5,all_evaluations_reused_development=True));r.write(HERE/'prefix-frozen-policy.json',dict(selected=selected,thresholds=dict(onset=.5,ongoing=.25,verifier=.5),later_month_selection=False));models=fit(['october','november'],'prefix-full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=population(backend,pop,'full',models);r.write(args.output/'task2_onset_prefix_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
