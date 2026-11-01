"""Date-resampled agreement for whole-link advance/delay timing actions."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import r
import arrival_prefix as prefix
import arrival_context as context
from onset_timing_evaluate import link_delta

HERE=context.HERE/'committee';HERE.mkdir(exist_ok=True)

class Consensus:
    def __init__(self,models):self.models=models
    def predict(self,x,**kw):return np.min([m.predict(x,**kw) for m in self.models],axis=0)

def fit(months,backend,name):
    xs=[];ys=[];ws=[];ds=[];dates=[]
    for pop in months:
        meta,z,c=context.cache(backend,pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ids=c[f'{i}_ri'];known=[];labels=[]
            for row,link in enumerate(ids):
                new=z[f'{i}_base'].copy();new[:,link]=c[f'{i}_pred'][row].astype(float);delta=link_delta(z[f'{i}_y'],z[f'{i}_m'],z[f'{i}_base'],new,link)
                if delta is not None:known.append(row);labels.append(delta>0)
            if known:xs.append(c[f'{i}_Z'][known]);ys.extend(labels);ds.extend(c[f'{i}_direction'][known]);ws.extend([1/counts[e['panel']]/len(known)]*len(known));dates.extend([e['origin'][:10]]*len(known))
    x=np.concatenate(xs);y=np.array(ys);d=np.array(ds);w=np.array(ws);w/=w.mean();dates=np.array(dates);foldmap={day:i%3 for i,day in enumerate(sorted(set(dates)))};fold=np.array([foldmap[day] for day in dates]);out={};records=[]
    for direction in range(2):
        models=[]
        for k in range(3):
            take=(d==direction)&(fold!=k)
            if take.sum()<60 or len(np.unique(y[take]))<2:break
            params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31);model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{direction}-fold{k}.txt'));models.append(model);records.append(dict(direction=direction,fold=k,train_dates=sorted(set(dates[take])),excluded_dates=sorted(set(dates[fold==k])),rows=int(take.sum())))
        if len(models)==3:out[direction]=Consensus(models)
    r.write(HERE/f'{name}-training.json',dict(months=months,folds=records,decision='all three probability >=0.5; no threshold search',parent_event_threshold=.5,verifier_threshold=.5,event_existence_preserved=True,post_arrival_tail_changed=False,no_class_weight_boost=True));return out

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);prefix.cache=lambda pop,regime:context.cache(backend,pop,regime);models=fit(['october'],backend,'probe');internal=prefix.population(backend,'november','early',models);b=internal['base']['scores'];eligible=[k for k,v in internal.items() if k!='base' and v['scores']['onset_macro']>b['onset_macro'] and v['scores']['FP']<=b['FP'] and v['scores']['TP']>=b['TP'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'base';r.write(args.output/'task2_arrival_committee_internal.json',dict(reports=internal,selected=selected,later_month_selection=False,all_evaluations_reused_development=True));r.write(HERE/'frozen-policy.json',dict(selected=selected,thresholds=dict(onset=.5,ongoing=.25,verifier=.5),later_month_selection=False));models=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        reports[pop]=prefix.population(backend,pop,'full',models);r.write(args.output/'task2_arrival_committee_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP','FN']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
