"""Partial-label three-way risk learning from frozen OOF rescue acceptance."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r,softmax,partial_loss
import shadow_rescue_precision as base
import shadow_rescue_trajectory as traj
import rescue_explanations as exp
import same_cause_views as same
HERE=ROOT/'work/task2-cascade/partial-explanation-cache';HERE.mkdir(exist_ok=True)

def allowed_states(y,m,h):
    y,m=np.asarray(y,bool),np.asarray(m,bool)
    if m[h] and y[h]:return np.array([1,0,0],bool)
    later=(y[h+1:]&m[h+1:]).any();suffix_known=m[h+1:].all()
    if m[h]:return np.array([0,1,0],bool) if later else np.array([0,0,1],bool) if suffix_known else np.array([0,1,1],bool)
    return np.array([1,1,0],bool) if later else np.array([1,0,1],bool) if suffix_known else np.ones(3,bool)

class Probabilities:
    def __init__(self,model):self.model=model
    def predict(self,x,num_threads=4):return softmax(self.model.predict(x,raw_score=True,num_threads=num_threads))

def rich(legacy,margin,raw,h,view):return traj.enriched(legacy,margin,raw,h,'observed')

def fit(months,backend,name):
    accepted,fold,teachers=base.oof_models(name);traj.LOOKUP.clear();records=[];stats=Counter()
    for pop in months:
        traj.prepare(pop,'early',name,True);meta,z=base.load(pop,'early');counts=Counter(e['panel'] for e in meta);windows={};requests=[]
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            x=z[f'{i}_Z'];score=teachers[fold[e['origin'][:10]]].predict(x,num_threads=4)-x[:,3] if len(x) else np.zeros(0);a=accepted[f'{pop}_{i}'];assert np.array_equal(a,score<=0);selected=np.flatnonzero(a);windows[i]=(selected,score);requests.extend(dict(i=i,link=int(z[f'{i}_ri'][j])) for j in selected)
        raw=base.research.raw_cache(backend,pop,'early',meta,requests)
        for i,(selected,score) in windows.items():
            e=meta[i];rows=[]
            for j in selected:
                link,h=int(z[f'{i}_ri'][j]),int(z[f'{i}_hi'][j]);a=allowed_states(z[f'{i}_y'][:,link],z[f'{i}_m'][:,link],h);stats[''.join(map(str,a.astype(int)))]+=1
                if a.all():continue
                x=rich(z[f'{i}_Z'][j],score[j],raw[f'{i}_{link}'],h,'observed');rows.append(dict(x=x,a=a,date=e['origin'][:10]))
            for c in rows:c['w']=1/counts[e['panel']]/len(rows)
            records.extend(rows)
    x=np.array([c['x'] for c in records]);a=np.array([c['a'] for c in records]);w=np.array([c['w'] for c in records]);w/=w.mean();dates=np.array([c['date'] for c in records]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);bags=[]
    for k in [-1,0,1,2]:
        tr=np.ones(len(x),bool) if k==-1 else fold!=k;aa,ww=a[tr],w[tr]
        def objective(raw,ds):
            _,g,h=partial_loss(raw,aa,ww);return g.astype('float32'),h.astype('float32')
        params=dict(objective=objective,num_class=3,num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
        m=lgb.train(params,lgb.Dataset(x[tr],label=np.zeros(tr.sum())),num_boost_round=160);m.save_model(str(HERE/f'{name}-{k}.txt'));bags.append(Probabilities(m))
    r.write(HERE/f'{name}-training.json',dict(months=months,informative_cells=len(records),allowed_target_counts=dict(stats),class_weights=False,threshold=.5,OOF_rescue=True,unknown_not_clear=True,ancestors_not_nested=True))
    compact=[lgb.Booster(model_file=str(exp.HERE/f'{name}-legacy-{k}.txt')) for k in [-1,0,1,2]]
    return dict(observed_binary=[exp.Explanation(m) for m in bags],legacy_binary=[same.SharedCause(c,m) for c,m in zip(compact,bags)])

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();protocol=r.read(args.output/'task2_origin_stress_frozen.json');base.check_models(protocol);backend=r.setup(args);base.HERE=HERE;base.risk_features=rich
    probe=fit(['october'],backend,'probe');traj.LOOKUP.clear();traj.prepare('november','early','probe');internal=base.evaluate(backend,'november','early','probe',probe);r.write(args.output/'task2_partial_explanations_internal.json',dict(reports=internal,shadow_only=True));print(json.dumps(dict(pop='november',scores={k:{j:v['scores'][j] for j in ['TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        traj.LOOKUP.clear();traj.prepare(pop,'full','full');reports[pop]=base.evaluate(backend,pop,'full','full',full)
        r.write(args.output/'task2_partial_explanations_external.json',dict(reports=reports,policy_names=dict(observed_binary='partial_three_way',legacy_binary='compact_and_partial_same_cause'),official_truth_available=False,repeated_development=True,default_unchanged=True,shadow_only=True))
        print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    base.check_models(protocol)
if __name__=='__main__':main()
