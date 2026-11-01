"""Predict whether a proposed continuous episode improves whole-window IoU.

Labels use training outcomes only. Inference uses causal candidate features.
Fixed .5 classification; no label-class weight boost or threshold search.
"""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import lightgbm as lgb
import run as cascade
from core import ROOT,r
from joint_experiment import merge,HERE

def episodes(ri,hi):
    for link in np.unique(ri):
        rows=np.flatnonzero(ri==link)
        for part in np.split(rows,np.flatnonzero(np.diff(hi[rows])>1)+1):
            if len(part):yield part

def gain(y,m,base,ri,hi):
    y=y.astype(bool);base=base>=.5;known=m[hi,ri]
    if not known.any():return None
    inter=(base&y&m).sum();union=((base|y)&m).sum()
    tp=(y[hi,ri]&known).sum();fp=(~y[hi,ri]&known).sum()
    before=inter/union if union else 1.
    after=(inter+tp)/(union+fp) if union+fp else 1.
    return float(after-before)

def episode_matrix(z,i,variant='directed_joint'):
    x=z[f'{i}_Z'];ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];groups=list(episodes(ri,hi));out=[]
    # The physical features are constant across an episode except horizon
    # interactions; retain their joint values once, pool temporal probabilities
    # to describe duration, onset, ending and evidence throughout the episode.
    for rows in groups:
        a=x[rows]
        out.append(np.r_[a.mean(0),a[:,103:122].min(0),a[:,103:122].max(0),len(rows)/6,hi[rows].min()/5,hi[rows].max()/5,hi[rows].mean()/5])
    width=x.shape[1]+42
    return np.array(out,dtype='float32').reshape(-1,width),groups

def load(pop,regime):
    meta=r.read(HERE/f'{regime}-{pop}.json');z=dict(np.load(HERE/f'{regime}-{pop}.npz'));j=dict(np.load(HERE/f'joint-{regime}-{pop}.npz'));schema=r.read(HERE/'joint-schema.json')
    return meta,merge(z,j,'directed_joint',schema)

def fit(months,name):
    xs=[];ys=[];ws=[];stats=Counter()
    for pop in months:
        meta,z=load(pop,'early');counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            xx,groups=episode_matrix(z,i);yy=[];valid=[]
            for j,rows in enumerate(groups):
                value=gain(z[f'{i}_y'],z[f'{i}_m'],z[f'{i}_base'],z[f'{i}_ri'][rows],z[f'{i}_hi'][rows])
                if value is None:continue
                # Zero utility is a negative label: adding unknown-only cells
                # is excluded, adding known-neutral fragments is not rewarded.
                valid.append(j);yy.append(value>0);stats['positive' if value>0 else 'nonpositive']+=1
            if valid:
                xs.append(xx[valid]);ys.extend(yy);ws.extend([1/counts[e['panel']]/len(valid)]*len(valid))
    x=np.concatenate(xs);y=np.array(ys);w=np.array(ws);w/=w.mean()
    params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
    model=lgb.train(params,lgb.Dataset(x,label=y,weight=w),num_boost_round=160);model.save_model(str(HERE/(name+'.txt')))
    r.write(HERE/(name+'.json'),dict(months=months,rows=len(y),labels=dict(stats),features=x.shape[1],threshold=.5,class_weight_boost=False,weights='equal candidate-window mass per road',target='strictly positive exact marginal proxy IoU of whole episode'))
    return model

def predict(z,meta,model):
    zz=dict(z)
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        x,groups=episode_matrix(z,i);decisions=model.predict(x,num_threads=4)>=.5 if len(x) else []
        keep=np.zeros(len(z[f'{i}_ri']))
        for rows,decision in zip(groups,decisions):keep[rows]=float(decision)
        zz[f'{i}_Z']=keep[:,None]
    class Identity:
        def predict(self,x,**kw):return x[:,0]
    return zz,Identity()

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);probe=fit(['october'],'episode-probe');meta,z=load('november','early');zz,identity=predict(z,meta,probe);result=cascade.metrics(meta,zz,backend,'verified',identity)[0]
    prior=r.read(args.output/'task2_joint_pattern_internal.json')['internal']['previous']
    selected=result['onset_macro']>prior['onset_macro'] and result['FP']<=prior['FP'] and result['negative_false_any']<=prior['negative_false_any']
    r.write(args.output/'task2_episode_internal.json',dict(scores=result,selected=selected,threshold=.5,selection='November only; improve IoU without increasing FP cells or negative false-any vs previous verifier'))
    r.write(HERE/'episode-frozen-policy.json',dict(selected=selected,threshold=.5,later_month_selection=False));model=fit(['october','november'],'episode-full');reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z=load(pop,'full');zz,identity=predict(z,meta,model);scores,rows=cascade.metrics(meta,zz,backend,'verified',identity);reports[pop]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_episode_external.json',dict(reports=reports,selected_on_november=selected,threshold=.5,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:scores[k] for k in ['onset_macro','TP','FP','FN','negative_false_any']})),flush=True)

if __name__=='__main__':main()
