"""Whole-link timing actions; event existence retained and thresholds fixed."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import lightgbm as lgb
from core import ROOT,r
from onset_phase import HERE,BASE
from episodes import load as old_load
from margin import Margin

def link_delta(y,m,b,new,link):
    old=b>=.5;pred=old.copy();pred[:,link]=new[:,link]>=.5;y=y.astype(bool)
    def iou(a):
        i=(a&y&m).sum();u=((a|y)&m).sum();return i/u if u else 1.
    known=((old[:,link]!=pred[:,link])&m[:,link]).any()
    return iou(pred)-iou(old) if known else None

def fit(months,name):
    xs=[];ys=[];ws=[];deltas=[]
    for pop in months:
        meta=r.read(HERE/f'early-{pop}.json');z=np.load(HERE/f'early-{pop}.npz');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ids=z[f'{i}_ri'];known=[];labels=[];ds=[]
            for row,link in enumerate(ids):
                value=link_delta(z[f'{i}_y'],z[f'{i}_m'],z[f'{i}_base'],z[f'{i}_shaped'],link)
                if value is not None:known.append(row);labels.append(value>0);ds.append(value)
            if known:xs.append(z[f'{i}_Z'][known]);ys.extend(labels);deltas.extend(ds);ws.extend([1/counts[e['panel']]/len(known)]*len(known))
    x=np.concatenate(xs);y=np.array(ys);w=np.array(ws);w/=w.mean();params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x,label=y,weight=w),num_boost_round=160);model.save_model(str(HERE/f'{name}.txt'));r.write(HERE/f'{name}.json',dict(months=months,link_actions=len(y),positive_utility=int(y.sum()),features=x.shape[1],threshold=.5,target='strictly positive exact marginal whole-link timing IoU',weights='equal proposed-window mass per road; no class boost',neutral_is_negative=True));return model

def evaluate(meta,z,old,backend,variant,gate,margin):
    rows=[];neg=defaultdict(list);TP=FP=FN=0;changes=0;events_lost=events_added=0;actions=0
    for i,e in enumerate(meta):
        b=z[f'{i}_base'];p=b>=(.5 if e['condition']==0 else .25);y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
        if e['condition']==0:
            before=p.copy()
            if variant in ['arrival_shape','gated_shape','gated_shape_margin']:
                ids=z[f'{i}_ri'];take=np.ones(len(ids),bool) if variant=='arrival_shape' else (gate.predict(z[f'{i}_Z'],num_threads=4)>=.5 if len(ids) else np.zeros(0,bool));ids=ids[take];p[:,ids]=z[f'{i}_shaped'][:,ids]>=.5;actions+=len(ids)
            if variant=='conditional_rescue':p|=z[f'{i}_conditional']>=.5
            if variant=='joint_rescue':p|=z[f'{i}_joint']>=.5
            if variant in ['margin','gated_shape_margin']:
                ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];x=old[f'{i}_Z'];accept=margin.predict(x,num_threads=4)>=.5 if len(x) else np.zeros(0,bool);p[hi[accept],ri[accept]]=True
            events_lost+=int((before.any(0)&~p.any(0)).sum());events_added+=int((~before.any(0)&p.any(0)).sum());changes+=int(((before!=p)&m).sum())
        if not e['has_future_queue']:
            if e['condition']==0:neg[e['panel']].append(bool((p&m).any()))
            continue
        rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
        if e['condition']==0:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
    result=backend[5](rows);result.update(onset_macro=float(np.mean([v for k,v in result['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None,known_changed_cells=changes,retimed_link_actions=actions,parent_events_deleted=events_lost,parent_events_added=events_added)
    return result,rows

def run_population(args,backend,pop,regime,gate):
    meta=r.read(HERE/f'{regime}-{pop}.json');z=dict(np.load(HERE/f'{regime}-{pop}.npz'));_,old=old_load(pop,regime);margin=Margin(lgb.Booster(model_file=str(BASE/('margin-probe.txt' if regime=='early' else 'margin-full.txt'))));variants=['base','margin','conditional_rescue','joint_rescue','arrival_shape','gated_shape','gated_shape_margin'];results={}
    for k in variants:
        scores,rows=evaluate(meta,z,old,backend,k,gate,margin);results[k]=dict(scores=scores,windows=rows)
    return results

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);gate=fit(['october'],'timing-probe');internal=run_population(args,backend,'november','early',gate);b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k!='base' and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_onset_timing_internal.json',dict(reports=internal,selected=selected,thresholds=dict(onset=.5,ongoing=.25,timing_verifier=.5),later_month_selection=False,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,thresholds=dict(onset=.5,ongoing=.25,timing_verifier=.5),later_month_selection=False));gate=fit(['october','november'],'timing-full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=run_population(args,backend,pop,'full',gate);r.write(args.output/'task2_onset_timing_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN','negative_false_any','parent_events_deleted']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
