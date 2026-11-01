"""Censoring-aware ordinal arrival evidence across all five adjacent horizons."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import r
import arrival_wait as wait
import arrival_unit as unit

HERE=wait.HERE/'ordinal';HERE.mkdir(exist_ok=True)

def ordinal_target(y,m,h):
    # An observed queued cell proves arrival, even if earlier cells are missing.
    # No observed positives does NOT prove absence if a prefix cell is missing.
    if (np.asarray(y[:h+1],bool)&np.asarray(m[:h+1],bool)).any():return 0
    if np.asarray(m[:h+1],bool).all():return 1
    return None

def fit(months,backend,name):
    samples=[]
    for pop in months:
        meta,z,protected,records=wait.candidates(backend,pop,'early',name,True,retain_unknown=True)
        for c in records:
            i,link=c['i'],c['link'];e=meta[i];temp=[]
            for h in range(5):
                y=ordinal_target(z[f'{i}_y'][:,link],z[f'{i}_m'][:,link],h)
                if y is None:continue
                x=unit.unit_features(c['x'][:-31],z[f'{i}_base'][:,link],protected[i][:,link],z[f'{i}_first'][link],z[f'{i}_before'][link],z[f'{i}_after'][link],z[f'{i}_conditional'][:,link],z[f'{i}_joint'][:,link],h,True)
                f=wait.timing_features(x,z[f'{i}_base'],z[f'{i}_first'],link,h,backend[3][e['panel']]['edges'])
                temp.append(dict(f=f,y=y,date=c['date'],w=c['w']))
            for v in temp:v['w']/=len(temp)
            samples.extend(temp)
    x=np.array([v['f'] for v in samples]);y=np.array([v['y'] for v in samples]);w=np.array([v['w'] for v in samples]);w/=w.mean();dates=np.array([v['date'] for v in samples]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);out=[]
    params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
    for k in [-1,0,1,2]:
        take=np.ones(len(y),bool) if k==-1 else fold!=k
        model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'{name}-{k}.txt'));out.append(model)
    r.write(HERE/f'{name}-training.json',dict(months=months,rows=len(y),classes=Counter(map(int,y)),features=x.shape[1],target='1: all prefix observed clear; 0: at least one observed queued prefix cell; ambiguous excluded',threshold=.5,all_horizons_per_link_weight_normalized=True,no_class_boost=True,params=params))
    return out

def evaluate(backend,pop,regime,name,models):
    # Reuse the monotone evaluator; adapt the output probabilities so its fixed
    # wait-veto API evaluates single and conservative date-bag consensus heads.
    class Multi:
        def predict(self,x,**kw):
            p=models[0].predict(x,num_threads=4);return np.column_stack([1-p,p,np.zeros(len(x))])
    return wait.evaluate(backend,pop,regime,name,dict(multi=Multi(),pair=models[0],pair_consensus=models[1:]))

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],backend,'probe')
    previous=wait.HERE;wait.HERE=HERE
    try:
        internal=evaluate(backend,'november','early','probe',probe);control=internal['committee_advance']['scores'];eligible=[k for k in ['multi','pair_consensus'] if internal[k]['scores']['TP']>=control['TP'] and internal[k]['scores']['FP']<control['FP'] and internal[k]['scores']['onset_macro']>=control['onset_macro']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'committee_advance'
        r.write(args.output/'task2_ordinal_wait_internal.json',dict(reports=internal,selected=selected,all_months_development=True));print(json.dumps(dict(pop='november',selected=selected,scores={k:{j:v['scores'][j] for j in ['onset_macro','TP','FP']} for k,v in internal.items()})),flush=True)
        full=fit(['october','november'],backend,'full');reports={}
        for pop in ['december','january','february_new','february_old']:
            reports[pop]=evaluate(backend,pop,'full','full',full);r.write(args.output/'task2_ordinal_wait_external.json',dict(reports=reports,selected=selected,all_months_development=True,official_truth_available=False,formal_R127_unchanged=True));print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','onset_macro','TP','FP']} for k,v in reports[pop].items()})),flush=True)
    finally:wait.HERE=previous

if __name__=='__main__':main()
