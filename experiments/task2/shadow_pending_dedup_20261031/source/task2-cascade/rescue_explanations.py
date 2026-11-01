"""Three-way rescue timing head: current queue / pending / clear suffix.
Veto only when an individual non-current explanation has probability >= .5.
No class boosts, changed cutoffs, identity rules, or future inference inputs.
"""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import shadow_rescue_precision as base
import shadow_rescue_trajectory as traj

HERE=traj.HERE/'timing_explanations';HERE.mkdir(exist_ok=True)

def timing_target(y,m,h):
    y=np.asarray(y,bool);m=np.asarray(m,bool)
    if not m[h]:return None
    if y[h]:return 0
    if (y[h+1:]&m[h+1:]).any():return 1
    return 2 if m[h:].all() else None

class Explanation:
    def __init__(self,model):self.model=model
    def predict(self,x,num_threads=4):
        p=self.model.predict(x,num_threads=num_threads)
        return p[:,1:].max(1)

def fit(months,backend,name):
    traj.LOOKUP.clear();targets={};counts=Counter()
    for pop in months:
        traj.prepare(pop,'early',name,True);meta,z=base.load(pop,'early');accepted,_,_=base.oof_models(name)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            for j in np.flatnonzero(accepted[f'{pop}_{i}']):
                link,h=int(z[f'{i}_ri'][j]),int(z[f'{i}_hi'][j]);key=np.asarray(z[f'{i}_Z'][j],dtype='float32').tobytes();t=timing_target(z[f'{i}_y'][:,link],z[f'{i}_m'][:,link],h)
                if key in targets:assert targets[key]==t
                targets[key]=t
    records=base.training(months,backend,name);rows=[]
    for c in records:
        key=np.asarray(c['views']['legacy'][:218],dtype='float32').tobytes();t=targets[key];counts[str(t)]+=1
        if t is not None:rows.append(dict(c,target=t))
    dates=np.array([c['date'] for c in rows]);mapping={d:i%3 for i,d in enumerate(sorted(set(dates)))};fold=np.array([mapping[d] for d in dates]);w=np.array([c['w'] for c in rows]);w/=w.mean();y=np.array([c['target'] for c in rows]);out={}
    for view in ['legacy','observed']:
        x=np.array([c['views'][view] for c in rows]);bags=[]
        for k in [-1,0,1,2]:
            tr=np.ones(len(y),bool) if k==-1 else fold!=k
            params=dict(objective='multiclass',num_class=3,num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=20,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261030,deterministic=True,force_col_wise=True,max_bin=31)
            m=lgb.train(params,lgb.Dataset(x[tr],label=y[tr],weight=w[tr]),num_boost_round=160);m.save_model(str(HERE/f'{name}-{view}-{k}.txt'));bags.append(Explanation(m))
        out[view+'_binary']=bags
    r.write(HERE/f'{name}-training.json',dict(months=months,cells=len(rows),targets=dict(counts),classes=['current_queue','pending_arrival','clear_remaining_suffix'],inference='max(P(pending),P(clear_suffix)) >= .5',class_weights=False,OOF_acceptance=True,unknown_targets_excluded=True,threshold=.5,shadow_only=True))
    return out

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();protocol=r.read(args.output/'task2_origin_stress_frozen.json');base.check_models(protocol);backend=r.setup(args);base.HERE=HERE;base.risk_features=traj.enriched
    probe=fit(['october'],backend,'probe');traj.LOOKUP.clear();traj.prepare('november','early','probe');internal=base.evaluate(backend,'november','early','probe',probe);b=internal['retained']['scores'];eligible=[k for k,v in internal.items() if k!='retained' and v['scores']['lost_retained_TP']==0 and v['scores']['all_onset_FP']<b['all_onset_FP'] and v['scores']['onset_macro']>=b['onset_macro'] and v['scores']['negative_false_any']<=b['negative_false_any']];selected=max(eligible,key=lambda k:internal[k]['scores']['onset_macro']) if eligible else 'retained'
    r.write(args.output/'task2_rescue_explanations_internal.json',dict(reports=internal,selected=selected,shadow_only=True));print(json.dumps(dict(pop='november',selected=selected,scores={k:{j:v['scores'][j] for j in ['TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in internal.items()})),flush=True)
    full=fit(['october','november'],backend,'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        traj.LOOKUP.clear();traj.prepare(pop,'full','full');reports[pop]=base.evaluate(backend,pop,'full','full',full)
        r.write(args.output/'task2_rescue_explanations_external.json',dict(reports=reports,selected=selected,shadow_only=True,official_truth_available=False,repeated_development_months=True,default_unchanged=True))
        print(json.dumps(dict(pop=pop,scores={k:{j:v['scores'][j] for j in ['proxy_IoU','TP','FP','lost_retained_TP','removed_rescue_FP']} for k,v in reports[pop].items()})),flush=True)
    base.check_models(protocol)
if __name__=='__main__':main()
