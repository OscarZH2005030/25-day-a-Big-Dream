"""Ablations selected on November only, before inspecting later results."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
import run as cascade
from core import ROOT,r,Context
from joint import joint_features

HERE=Path(__file__).resolve().parent

def augment(args,backend,regime,pop):
    path=HERE/f'joint-{regime}-{pop}.npz'
    if path.exists():return dict(np.load(path))
    meta=r.read(HERE/f'{regime}-{pop}.json');z=dict(np.load(HERE/f'{regime}-{pop}.npz'));out={};checks=[]
    for i,e in enumerate(meta):
        if e['condition']!=0:continue
        ri=z[f'{i}_ri'];hi=z[f'{i}_hi']
        if len(ri):
            ee=dict(e,origin_slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5);s=backend[1](ee);a,names,local=joint_features(s);out[f'{i}_J']=a[ri,hi]
            if e['panel'] not in checks:
                ss=dict(s,y=1-s['y'],m=~s['m'],has_future_queue=not s['has_future_queue']);aa,_,_=joint_features(ss);assert np.array_equal(a,aa);checks.append(e['panel'])
        else:out[f'{i}_J']=np.zeros((0,96),dtype='float32') # schema assigned below
        if (i+1)%128==0:print(json.dumps(dict(phase='joint_features',pop=pop,done=i+1,total=len(meta))),flush=True)
    width=len(names)
    for k,v in out.items():
        if len(v)==0:out[k]=np.zeros((0,width),dtype='float32')
    np.savez_compressed(path,**out);r.write(HERE/'joint-schema.json',dict(names=names,local_columns=local,columns=width,target_mutation_checks=checks,validity='simultaneous channel quality and eligibility masks; no forward carry'))
    return out

def merge(z,j,variant,schema):
    zz=dict(z)
    for key,v in j.items():
        if not key.endswith('_J'):continue
        prefix=key[:-2];old=z[prefix+'_Z']
        zz[prefix+'_Z']=np.column_stack([old,v[:,:schema['local_columns']] if variant=='local' else v])
    return zz

def fit(months,variant,name,schema):
    xx=[];yy=[];ww=[]
    for pop in months:
        meta=r.read(HERE/f'early-{pop}.json');z=dict(np.load(HERE/f'early-{pop}.npz'));j=dict(np.load(HERE/f'joint-early-{pop}.npz'));z=merge(z,j,variant,schema);counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            if e['condition']!=0:continue
            ri=z[f'{i}_ri'];hi=z[f'{i}_hi'];known=z[f'{i}_m'][hi,ri]
            if not known.any():continue
            xx.append(z[f'{i}_Z'][known]);yy.append(z[f'{i}_y'][hi[known],ri[known]]);ww.append(np.full(known.sum(),1/counts[e['panel']]/known.sum()))
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean()
    params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=30,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
    model=lgb.train(params,lgb.Dataset(x,label=y,weight=w),num_boost_round=160);model.save_model(str(HERE/(name+'.txt')))
    r.write(HERE/(name+'.json'),dict(months=months,variant=variant,threshold=.5,features=x.shape[1],rows=len(y),class_weight_boost=False,parameters=params))
    return model

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    for pop in ['october','november']:augment(args,backend,'early',pop)
    schema=r.read(HERE/'joint-schema.json');meta=r.read(HERE/'early-november.json');z=dict(np.load(HERE/'early-november.npz'));j=dict(np.load(HERE/'joint-early-november.npz'))
    previous=lgb.Booster(model_file=str(HERE/'verifier-probe.txt'));internal={'base':cascade.metrics(meta,z,backend,'base')[0],'previous':cascade.metrics(meta,z,backend,'verified',previous)[0]}
    for variant in ['local','directed_joint']:
        model=fit(['october'],variant,'joint-probe-'+variant,schema);zz=merge(z,j,variant,schema);internal[variant]=cascade.metrics(meta,zz,backend,'verified',model)[0]
    # Freeze selection using November onset IoU, with no increase in negative
    # false-any or false-positive cells relative to the previous verifier.
    prior=internal['previous'];eligible=[v for v in ['local','directed_joint'] if internal[v]['onset_macro']>prior['onset_macro'] and internal[v]['FP']<=prior['FP'] and internal[v]['negative_false_any']<=prior['negative_false_any']]
    selected=max(eligible,key=lambda v:internal[v]['onset_macro']) if eligible else 'previous'
    policy=dict(selected=selected,selection='November only: improve onset IoU, no more FP cells or negative false-any than previous verifier',threshold=.5,internal=internal,later_months_used_for_selection=False)
    r.write(args.output/'task2_joint_pattern_internal.json',policy);r.write(HERE/'joint-frozen-policy.json',dict(selected=selected,threshold=.5,later_month_selection=False))
    # Both feature ablations remain visible as research diagnostics. Only the
    # frozen selection may be called the selected policy.
    models={v:fit(['october','november'],v,'joint-full-'+v,schema) for v in ['local','directed_joint']};models['previous']=lgb.Booster(model_file=str(HERE/'verifier-full.txt'))
    results={}
    for pop in ['december','january','february_new','february_old']:
        j=augment(args,backend,'full',pop);meta=r.read(HERE/f'full-{pop}.json');z=dict(np.load(HERE/f'full-{pop}.npz'));scores={'base':cascade.metrics(meta,z,backend,'base')[0]}
        for variant,model in models.items():
            zz=z if variant=='previous' else merge(z,j,variant,schema);scores[variant]=cascade.metrics(meta,zz,backend,'verified',model)[0]
        results[pop]=scores;r.write(args.output/'task2_joint_pattern_external.json',dict(reports=results,frozen_selection=selected,threshold=.5,all_evaluations_reused_development=True,official_truth_available=False))
        print(json.dumps(dict(population=pop,selected=selected,scores={k:{p:v.get(p) for p in ['onset_macro','TP','FP','FN','negative_false_any']} for k,v in scores.items()})),flush=True)

if __name__=='__main__':main()
