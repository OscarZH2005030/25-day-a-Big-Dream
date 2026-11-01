"""One-cell arrival edits: preserve every other parent prediction exactly."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import arrival_context as context
from arrival_committee import Consensus
from arrival_safe import oof_rescues,Keep
from onset_phase import BASE
from episodes import load
from margin import Margin

HERE=context.HERE/'unit_edit';HERE.mkdir(exist_ok=True)
UNIT_EXTRA_WIDTH=31

def unit_proposal(b,prefix,direction):
    old=b>=.5;first=int(np.flatnonzero(old)[0])
    if direction==0:
        if first==0:return None
        h=first-1;value=True
    else:
        if old.sum()<2:return None
        h=first;value=False
    return h,value

def unit_features(x,b,rescue,first,before,after,conditional,joint,h,value):
    entropy=-np.sum(first*np.log(np.maximum(first,1e-12)))/np.log(6);sortedp=np.sort(first);earliest=int(np.flatnonzero(b>=.5)[0]);latest=int(np.flatnonzero(b>=.5)[-1]);prev=max(h-1,0);nxt=min(h+1,5);extra=np.r_[np.eye(6)[h],value,b[h],b[prev],b[nxt],b[nxt]-b[h],b[h]-b[prev],b.max(),first[h],first[:h+1].sum(),before[h],after[h],conditional[h],joint[h],entropy,sortedp[-1]-sortedp[-2],earliest/5,latest/5,rescue,rescue.any(),(b>=.5).mean()]
    assert len(extra)==UNIT_EXTRA_WIDTH
    return np.r_[x,extra].astype('float32')

def dataset(months,backend,name):
    rescues=oof_rescues(months,backend,name);xs=[];ys=[];ws=[];ds=[];dates=[];countsout=Counter()
    for pop in months:
        meta,z,c=context.cache(backend,pop,'early');_,old=load(pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];accept=rescues[f'{pop}_{i}'];rescue=np.zeros_like(z[f'{i}_base'],bool);rescue[hi[accept],ri[accept]]=True;ids=c[f'{i}_ri'];temp=[]
            for row,link in enumerate(ids):
                d=int(c[f'{i}_direction'][row]);proposal=unit_proposal(z[f'{i}_base'][:,link],c[f'{i}_pred'][row],d)
                if proposal is None:continue
                h,value=proposal
                if rescue[h,link] and value:countsout['duplicate_rescue_noop']+=1;continue
                if not z[f'{i}_m'][h,link]:countsout['unknown_label_excluded']+=1;continue
                # Restoring the frozen rescue makes an attempted removal of a
                # rescued cell a no-op, but rescue proposals never overlap a
                # parent positive with the existing candidate construction.
                if rescue[h,link] and not value:countsout['protected_rescue_noop']+=1;continue
                y=int(bool(z[f'{i}_y'][h,link])==value);x=unit_features(c[f'{i}_Z'][row],z[f'{i}_base'][:,link],rescue[:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,value);temp.append((x,y,d));countsout['correctable']+=y;countsout['known_unit_actions']+=1
            for x,y,d in temp:xs.append(x);ys.append(y);ds.append(d);ws.append(1/counts[e['panel']]/len(temp));dates.append(e['origin'][:10])
    r.write(HERE/f'{name}-data.json',dict(months=months,counts=dict(countsout),target='one changed known cell is correct; unknown excluded; no multi-cell utility tradeoffs',rescue_baseline='out-of-fold by global date'))
    return np.array(xs,dtype='float32'),np.array(ys),np.array(ds),np.array(ws),np.array(dates)

def fit(months,backend,name):
    x,y,d,w,dates=dataset(months,backend,name);w/=w.mean();mapping={day:i%3 for i,day in enumerate(sorted(set(dates)))};fold=np.array([mapping[day] for day in dates]);out={};records=[]
    for kind in ['single','committee']:
        bags=[]
        for k in ([-1] if kind=='single' else range(3)):
            bag={}
            for direction in range(2):
                take=(d==direction)&(np.ones(len(y),bool) if k==-1 else fold!=k)
                if take.sum()<60 or len(np.unique(y[take]))<2:bag[direction]=Keep();continue
                params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{kind}-{k}-{direction}.txt'));bag[direction]=model;records.append(dict(kind=kind,fold=k,direction=direction,rows=int(take.sum()),positive=int(y[take].sum())))
            bags.append(bag)
        out[kind]={direction:(bags[0][direction] if kind=='single' else Consensus([b[direction] for b in bags])) for direction in range(2)}
    r.write(HERE/f'{name}-training.json',dict(months=months,heads=records,features=x.shape[1],thresholds=dict(parent=.5,ongoing=.25,gate=.5),seed=20261030,max_changed_cells_per_link=1,inference_iteration=False,no_class_weight_boost=True))
    return out

def evaluate(backend,pop,regime,models,output_dir=None):
    meta,z,c=context.cache(backend,pop,regime);_,old=load(pop,regime);margin=Margin(lgb.Booster(model_file=str(BASE/('margin-probe.txt' if regime=='early' else 'margin-full.txt'))));results={};store={}
    for kind,heads in [('base',{}),('margin',{})]+list(models.items()):
        rows=[];neg={};TP=FP=FN=lost=actions=0
        for i,e in enumerate(meta):
            parent=z[f'{i}_base']>=(.5 if e['condition']==0 else .25);p=parent.copy();y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
            if e['condition']==0:
                rescue=np.zeros_like(parent);ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];xx=old[f'{i}_Z'];take=margin.predict(xx,num_threads=4)>=.5 if len(xx) else np.zeros(0,bool);rescue[hi[take],ri[take]]=True
                if kind!='base':p|=rescue
                if heads:
                    for row,link in enumerate(c[f'{i}_ri']):
                        d=int(c[f'{i}_direction'][row])
                        if d not in heads:continue
                        proposal=unit_proposal(z[f'{i}_base'][:,link],c[f'{i}_pred'][row],d)
                        if proposal is None:continue
                        h,value=proposal
                        if rescue[h,link]:continue
                        x=unit_features(c[f'{i}_Z'][row],z[f'{i}_base'][:,link],rescue[:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,value)
                        if heads[d].predict(x[None],num_threads=4)[0]>=.5:p[h,link]=value;actions+=1
                if e['has_future_queue']:lost+=int((parent&~p&y&m).sum())
            if not e['has_future_queue']:
                if e['condition']==0:neg.setdefault(e['panel'],[]).append(bool((p&m).any()))
                continue
            rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
            if e['condition']==0:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
            if kind=='committee' and e['condition']==0:store[f'{i}_pred']=p
            if kind=='committee_advance' and e['condition']==0:store[f'advance_{i}_pred']=p
            if kind=='conditional_advance' and e['condition']==0:store[f'conditional_{i}_pred']=p
        s=backend[5](rows);s.update(onset_macro=float(np.mean([v for k,v in s['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None,lost_original_R127_TP=lost,accepted_unit_actions=actions);results[kind]=dict(scores=s,windows=rows)
    directory=HERE if output_dir is None else Path(output_dir);directory.mkdir(exist_ok=True);dest=directory/f'pred-{regime}-{pop}.npz';saved=dict(np.load(dest)) if dest.exists() else {};saved.update(store);np.savez_compressed(dest,**saved);return results

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early',probe);b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k not in ['base','margin'] and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_arrival_unit_internal.json',dict(reports=internal,selected=selected,later_month_selection=False,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,thresholds=dict(parent=.5,ongoing=.25,gate=.5),later_month_selection=False));models=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full',models);r.write(args.output/'task2_arrival_unit_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN','lost_original_R127_TP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
