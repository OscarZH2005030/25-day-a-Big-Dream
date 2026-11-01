"""Continuous-speed verification only for newly advanced cells.

Every parent/rescue positive is hard preserved. New validators use the fixed
published speed boundary; no classification or physical threshold search.
"""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import arrival_context as context
from arrival_unit import unit_proposal,unit_features
from arrival_unit_directional import models as load_unit
from arrival_safe import oof_rescues
from onset_phase import BASE
from episodes import load
from margin import Margin

HERE=context.HERE/'advance_precision';HERE.mkdir(exist_ok=True)

def training(months,backend,name):
    rescues=oof_rescues(months,backend,name);xs=[];ys=[];ws=[];dates=[]
    for pop in months:
        meta,z,c=context.cache(backend,pop,'early');_,old=load(pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];accept=rescues[f'{pop}_{i}'];rescue=np.zeros_like(z[f'{i}_base'],bool);rescue[hi[accept],ri[accept]]=True;temp=[];raw=None
            for row,link in enumerate(c[f'{i}_ri']):
                if c[f'{i}_direction'][row]!=0:continue
                proposal=unit_proposal(z[f'{i}_base'][:,link],c[f'{i}_pred'][row],0)
                if proposal is None:continue
                h,value=proposal
                if rescue[h,link] or not z[f'{i}_m'][h,link]:continue
                if raw is None:
                    slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;raw=backend[1].load_day(e['panel'],e['origin'][:10]);net=backend[3][e['panel']]
                x=unit_features(c[f'{i}_Z'][row],z[f'{i}_base'][:,link],rescue[:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,value);margin=(raw['speed'][slot+h+1,link]-net['threshold'][link])/net['free_speed'][link];assert (margin<=0)==bool(z[f'{i}_y'][h,link]);temp.append((x,margin+x[3]))
            for x,y in temp:xs.append(x);ys.append(y);ws.append(1/counts[e['panel']]/len(temp));dates.append(e['origin'][:10])
    return np.array(xs,dtype='float32'),np.array(ys),np.array(ws),np.array(dates)

def fit(months,backend,name):
    x,y,w,dates=training(months,backend,name);w/=w.mean();mapping={day:i%3 for i,day in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);out={};provenance=[]
    for kind in ['single','committee']:
        models=[]
        for k in ([-1] if kind=='single' else range(3)):
            take=np.ones(len(y),bool) if k==-1 else fold!=k;params=dict(objective='regression',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{kind}-{k}.txt'));models.append(model);provenance.append(dict(kind=kind,fold=k,rows=int(take.sum()),train_dates=sorted(set(dates[take])),excluded_dates=sorted(set(dates[~take]))))
        out[kind]=models
    r.write(HERE/f'{name}-training.json',dict(months=months,rows=len(y),features=x.shape[1],target='future normalized speed distance to published queue line, residualized by latest observed margin',thresholds=dict(parent=.5,ongoing=.25,binary_validator=.5,predicted_speed_margin=0),physical_threshold_search=False,no_class_weight_boost=True,folds=provenance));return out

def physical_accept(models,x):
    # Reject disagreement between independently date-resampled regressors.
    predicted=np.array([model.predict(x,num_threads=4)-x[:,3] for model in models])
    return (predicted<=0).all(0),predicted

def evaluate(backend,pop,regime,physical):
    meta,z,c=context.cache(backend,pop,regime);_,old=load(pop,regime);unit=load_unit('probe' if regime=='early' else 'full')['committee_advance'][0];rescue_model=Margin(lgb.Booster(model_file=str(BASE/('margin-probe.txt' if regime=='early' else 'margin-full.txt'))));reports={};store={};case_catalog=[]
    for variant in ['base','margin','unit_advance','speed_single','speed_committee']:
        rows=[];neg={};TP=FP=FN=lost=actions=0
        for i,e in enumerate(meta):
            parent=z[f'{i}_base']>=(.5 if e['condition']==0 else .25);p=parent.copy();y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
            if e['condition']==0:
                rescue=np.zeros_like(parent);ri=old[f'{i}_ri'];hi=old[f'{i}_hi'];xx=old[f'{i}_Z'];take=rescue_model.predict(xx,num_threads=4)>=.5 if len(xx) else np.zeros(0,bool);rescue[hi[take],ri[take]]=True
                if variant!='base':p|=rescue
                if variant not in ['base','margin']:
                    for row,link in enumerate(c[f'{i}_ri']):
                        if c[f'{i}_direction'][row]!=0:continue
                        proposed=unit_proposal(z[f'{i}_base'][:,link],c[f'{i}_pred'][row],0)
                        if proposed is None:continue
                        h,value=proposed
                        if p[h,link]:continue
                        x=unit_features(c[f'{i}_Z'][row],z[f'{i}_base'][:,link],rescue[:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,value)[None]
                        if unit.predict(x,num_threads=4)[0]<.5:continue
                        accepted=True;predicted=None
                        if variant.startswith('speed_'):accepted,predicted=physical_accept(physical[variant.replace('speed_','')],x);accepted=bool(accepted[0])
                        if accepted:p[h,link]=True;actions+=1
                        if variant=='speed_committee' and e['has_future_queue'] and m[h,link]:case_catalog.append(dict(panel=e['panel'],origin=e['origin'],link_index=int(link),horizon_minutes=(int(h)+1)*5,true_queue=bool(y[h,link]),physical_accepted=accepted,predicted_normalized_margins=predicted[:,0].tolist()))
                assert not (parent&~p).any()
                if variant!='base':assert not (rescue&~p).any()
                if e['has_future_queue']:lost+=int((parent&~p&y&m).sum())
                if variant=='speed_committee':store[f'{i}_pred']=p
            if not e['has_future_queue']:
                if e['condition']==0:neg.setdefault(e['panel'],[]).append(bool((p&m).any()))
                continue
            rows.append(dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)))
            if e['condition']==0:TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
        s=backend[5](rows);s.update(onset_macro=float(np.mean([v for k,v in s['by_panel_condition'].items() if k.endswith('queue_onset')])),TP=TP,FP=FP,FN=FN,negative_false_any=float(np.mean([np.mean(v) for v in neg.values()])) if neg else None,lost_original_R127_TP=lost,accepted_advance_actions=actions);reports[variant]=dict(scores=s,windows=rows)
    np.savez_compressed(HERE/f'pred-{regime}-{pop}.npz',**store);r.write(HERE/f'cases-{regime}-{pop}.json',case_catalog);return reports

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],backend,'probe');internal=evaluate(backend,'november','early',probe);base=internal['margin']['scores'];control=internal['unit_advance']['scores'];eligible=[k for k in ['speed_single','speed_committee'] if internal[k]['scores']['onset_macro']>=base['onset_macro'] and internal[k]['scores']['FP']<control['FP'] and internal[k]['scores']['TP']>=base['TP'] and internal[k]['scores']['negative_false_any']<=control['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'unit_advance';r.write(args.output/'task2_advance_precision_internal.json',dict(reports=internal,selected_research_candidate=selected,formal_parent='R127 unchanged',selection_reference='retain margin baseline recall, reduce added FP vs unit advance, improve or tie margin onset',later_month_selection=False,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected_research_candidate=selected,formal_parent_unchanged=True,thresholds=dict(parent=.5,ongoing=.25,binary_validator=.5,predicted_speed_margin=0),later_month_selection=False));models=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=evaluate(backend,pop,'full',models);r.write(args.output/'task2_advance_precision_external.json',dict(reports=reports,frozen_research_selection=selected,formal_parent='R127 unchanged',all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected_research_candidate=selected,scores={k:{j:v['scores'][j] for j in ['onset_macro','TP','FP','lost_original_R127_TP']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
