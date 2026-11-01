"""Deployment-aligned arrival edits and certified Pareto-safe supervision.

Thresholds .5/.25/.5 are unchanged. Safety is learned as a target, not a
manually chosen confidence cutoff. Unknown changed labels are not safe zeros.
"""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import arrival_context as context
from arrival_committee import Consensus
from onset_phase import BASE
from episodes import load
from margin import Margin

HERE=context.HERE/'deployment_safe';HERE.mkdir(exist_ok=True)

def overlap_features(b,rescue,new,first,before,after):
    """Predictions only: no future truth or future quality in feature inputs."""
    parent=b>=.5;base=parent|rescue;post=new|rescue;oldfirst=int(np.flatnonzero(base)[0]);newfirst=int(np.flatnonzero(post)[0]);changed=base!=post;rawadded=new&~parent;mode=int(np.argmax(first));entropy=-np.sum(first*np.log(np.maximum(first,1e-12)))/np.log(6);sortedp=np.sort(first);mean=(first*np.arange(6)).sum();variance=(first*(np.arange(6)-mean)**2).sum()
    sums=np.r_[entropy,first.max(),sortedp[-1]-sortedp[-2],mean/5,np.sqrt(max(variance,0))/5,first[:oldfirst+1].sum(),first[:newfirst+1].sum(),first[oldfirst],first[newfirst],before[oldfirst],after[oldfirst],before[newfirst],after[newfirst],(newfirst-oldfirst)/5,mode/5,b[oldfirst],b.max(),base.mean(),post.mean(),rescue.mean(),(rawadded&rescue).sum()/6,changed.sum()/6,(~base&post).sum()/6,(base&~post).sum()/6]
    return np.r_[sums,base,post,rescue,post.astype(float)-base.astype(float)].astype('float32')

def action_labels(y,m,base,post):
    y=np.asarray(y,bool);m=np.asarray(m,bool);changed=base!=post
    if not changed.any():return 0,None,dict(noop=True,beneficial=0,harmful=0,unknown=0)
    beneficial=((~base&post&y)|(base&~post&~y))&m;harmful=((~base&post&~y)|(base&~post&y))&m;unknown=changed&~m
    safe=0 if harmful.any() else (1 if beneficial.any() and not unknown.any() else None)
    return safe,None,dict(noop=False,beneficial=int(beneficial.sum()),harmful=int(harmful.sum()),unknown=int(unknown.sum()))

def window_iou(y,p,m):
    u=((y|p)&m).sum();return float((y&p&m).sum()/u) if u else 1.

def deployed_delta(y,m,base,post):
    if not ((base!=post)&m).any():return 0. if np.array_equal(base,post) else None
    return window_iou(y,post,m)-window_iou(y,base,m)

def oof_rescues(months,backend,name):
    dest=HERE/f'{name}-oof-rescues.npz'
    if dest.exists():return dict(np.load(dest))
    xs=[];ts=[];ws=[];days=[];windows=[];cursor=0
    for pop in months:
        meta,z=load(pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];x=z[f'{i}_Z'];m=z[f'{i}_m'][hi,ri];target=np.zeros(len(x))
            if m.any():
                slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;net=backend[3][e['panel']];raw=backend[1].load_day(e['panel'],e['origin'][:10]);v=raw['speed'][slot+1:slot+7];target[m]=(v[hi[m],ri[m]]-net['threshold'][ri[m]])/net['free_speed'][ri[m]]+x[m,3]
            xs.append(x);ts.append(target);ws.append(np.where(m,1/counts[e['panel']]/max(m.sum(),1),0));days.extend([e['origin'][:10]]*len(x));windows.append((pop,i,cursor,cursor+len(x)));cursor+=len(x)
    x=np.concatenate(xs);target=np.concatenate(ts);w=np.concatenate(ws);days=np.array(days);mapping={d:i%3 for i,d in enumerate(sorted(set(days)))};fold=np.array([mapping[d] for d in days]);scores=np.zeros(len(x));records=[]
    for k in range(3):
        tr=(fold!=k)&(w>0);te=fold==k;ww=w[tr]/w[tr].mean();params=dict(objective='regression',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[tr],label=target[tr],weight=ww),num_boost_round=160);scores[te]=model.predict(x[te],num_threads=4)-x[te,3];model.save_model(str(HERE/f'{name}-rescue-fold{k}.txt'));assert not set(days[tr])&set(days[te]);records.append(dict(fold=k,train_dates=sorted(set(days[tr])),heldout_dates=sorted(set(days[te]))))
    out={f'{pop}_{i}':scores[start:end]<=0 for pop,i,start,end in windows};np.savez_compressed(dest,**out);r.write(HERE/f'{name}-rescue-provenance.json',dict(method='3-fold by global date; each window rescue predictions exclude all labels from that date',folds=records,months=months));return out

def rows(months,backend,name,regime='early'):
    oof=oof_rescues(months,backend,name) if regime=='early' else None;fixed=Margin(lgb.Booster(model_file=str(BASE/'margin-full.txt'))) if regime=='full' else None;xs=[];utility=[];safe=[];ds=[];weights=[];dates=[];stats=Counter()
    for pop in months:
        meta,z,c=context.cache(backend,pop,regime);_,old=load(pop,regime);counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];x=old[f'{i}_Z'];accept=oof[f'{pop}_{i}'] if oof is not None else (fixed.predict(x,num_threads=4)>=.5 if len(x) else np.zeros(0,bool));rescue=np.zeros_like(z[f'{i}_base'],bool);rescue[hi[accept],ri[accept]]=True;parent=z[f'{i}_base']>=.5;base=parent|rescue;y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m'];ids=c[f'{i}_ri'];temp=[]
            for row,link in enumerate(ids):
                post=base.copy();post[:,link]=c[f'{i}_pred'][row]|rescue[:,link];label,_,details=action_labels(y[:,link],m[:,link],base[:,link],post[:,link]);delta=deployed_delta(y,m,base,post);stats['safe_unknown']+=int(label is None);stats['safe_positive']+=int(label==1);stats['safe_negative']+=int(label==0);stats['noops']+=int(details['noop'])
                if delta is None and label is None:continue
                extra=overlap_features(z[f'{i}_base'][:,link],rescue[:,link],c[f'{i}_pred'][row],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link]);temp.append((np.r_[c[f'{i}_Z'][row],extra],int(delta>0) if delta is not None else -1,int(label) if label is not None else -1,c[f'{i}_direction'][row]))
            for x,u,s,d in temp:xs.append(x);utility.append(u);safe.append(s);ds.append(d);weights.append(1/counts[e['panel']]/len(temp));dates.append(e['origin'][:10])
    r.write(HERE/f'{name}-labels.json',dict(stats=dict(stats),source_months=months,regime=regime,positive_safety='all changed cells observed and every change beneficial; observed harm is negative; unknown with no observed harm excluded from safety fit',future_labels_in_features=False))
    return dict(X=np.array(xs,dtype='float32'),utility=np.array(utility),safe=np.array(safe),direction=np.array(ds),w=np.array(weights),date=np.array(dates))

class Keep:
    def predict(self,x,**kw):return np.zeros(len(x))

def fit(months,backend,name,regime='early'):
    data=rows(months,backend,name,regime);x=data['X'];d=data['direction'];w=data['w'];w/=w.mean();mapping={day:i%3 for i,day in enumerate(sorted(set(data['date'])))};fold=np.array([mapping[day] for day in data['date']]);out={};records=[]
    for objective,committee in [('utility',False),('safe',False),('safe',True)]:
        key=objective+('_committee' if committee else '_single');bags=[]
        for k in (range(3) if committee else [-1]):
            y=data[objective];included=(y>=0)&(np.ones(len(y),bool) if k==-1 else fold!=k);bag={}
            for direction in range(2):
                take=included&(d==direction)
                if take.sum()<60 or len(np.unique(y[take]))<2:bag[direction]=Keep();continue
                params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{key}-{k}-{direction}.txt'));bag[direction]=model;records.append(dict(kind=key,fold=k,direction=direction,rows=int(take.sum()),positive=int(y[take].sum())))
            bags.append(bag)
        out[key]={direction:(Consensus([b[direction] for b in bags]) if committee else bags[0][direction]) for direction in range(2)}
    r.write(HERE/f'{name}-training.json',dict(months=months,regime=regime,rows=len(x),features=x.shape[1],heads=records,thresholds=dict(parent=.5,ongoing=.25,gate=.5),seed=20261030,no_class_weight_boost=True,utility_baseline='parent OR frozen/out-of-fold margin rescue, matching deployed combination'))
    return out

def evaluate(backend,pop,regime,models):
    meta,z,c=context.cache(backend,pop,regime);_,old=load(pop,regime);margin=Margin(lgb.Booster(model_file=str(BASE/('margin-probe.txt' if regime=='early' else 'margin-full.txt'))));results={};predictions={}
    for kind,heads in [('base',{}),('margin',{})]+list(models.items()):
        rowsout=[];neg={};TP=FP=FN=0;lostTP=addedFP=0;actions=0
        for i,e in enumerate(meta):
            parent=z[f'{i}_base']>=(.5 if e['condition']==0 else .25);p=parent.copy();y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
            if e['condition']==0:
                rescue=np.zeros_like(parent);ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];xx=old[f'{i}_Z'];take=margin.predict(xx,num_threads=4)>=.5 if len(xx) else np.zeros(0,bool);rescue[hi[take],ri[take]]=True;baseline=parent|rescue
                if kind!='base':p|=rescue
                ids=c[f'{i}_ri']
                if heads:
                    for row,link in enumerate(ids):
                        extra=overlap_features(z[f'{i}_base'][:,link],rescue[:,link],c[f'{i}_pred'][row],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link]);x=np.r_[c[f'{i}_Z'][row],extra][None];model=heads[int(c[f'{i}_direction'][row])]
                        if model.predict(x,num_threads=4)[0]>=.5:p[:,link]=c[f'{i}_pred'][row]|rescue[:,link];actions+=1
                if e['has_future_queue']:lostTP+=int((parent&~p&y&m).sum());addedFP+=int((p&~baseline&~y&m).sum())
            if not e['has_future_queue']:
                if e['condition']==0:neg.setdefault(e['panel'],[]).append(bool((p&m).any()))
                continue
            rowsout.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
            if e['condition']==0:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
            if kind=='safe_committee' and e['condition']==0:predictions[f'{i}_pred']=p
        score=backend[5](rowsout);score.update(onset_macro=float(np.mean([v for k,v in score['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None,lost_original_R127_TP=lostTP,new_FP_outside_margin=addedFP,accepted_link_actions=actions);results[kind]=dict(scores=score,windows=rowsout)
    np.savez_compressed(HERE/f'pred-{regime}-{pop}.npz',**predictions);return results

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early',probe);b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k not in ['base','margin'] and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_arrival_safe_internal.json',dict(reports=internal,selected=selected,later_month_selection=False,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,thresholds=dict(parent=.5,ongoing=.25,gate=.5),later_month_selection=False));models=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full',models);r.write(args.output/'task2_arrival_safe_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN','lost_original_R127_TP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
