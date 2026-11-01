"""Shared vs phase-specific correction, with a predeclared core guard."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
from core import ROOT,r
import ongoing_gate as gate
from queue_phase import phase_features,PHASES,invariant_parent_features

HERE=gate.HERE

def load(pop,regime):
    meta=r.read(HERE/f'{regime}-{pop}.json');z=dict(np.load(HERE/f'{regime}-{pop}.npz'));c=dict(np.load(HERE/f'gate-{regime}-{pop}.npz'));return meta,z,c

def augment(backend,pop,regime):
    dest=HERE/f'phase-{regime}-{pop}.npz'
    if dest.exists():return dict(np.load(dest))
    meta,z,c=load(pop,regime);out={};checks=[];names=None
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        ri=c[f'{i}_ri'];hi=c[f'{i}_hi']
        if len(ri):
            slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));a,phase,names=phase_features(s,z[f'{i}_base']);out[f'{i}_A']=a[ri,hi];out[f'{i}_phase']=phase[ri]
            if e['panel'] not in checks:
                changed=dict(s,y=1-s['y'],m=~s['m'],speed_y=np.ones_like(s['y'])*999);aa,pp,_=phase_features(changed,z[f'{i}_base']);assert np.array_equal(a,aa) and np.array_equal(phase,pp);checks.append(e['panel'])
        else:out[f'{i}_A']=None;out[f'{i}_phase']=np.zeros(0,int)
        if (i+1)%512==0:print(json.dumps(dict(stage='phase_features',pop=pop,done=i+1)),flush=True)
    for k,v in out.items():
        if v is None:out[k]=np.zeros((0,len(names)),dtype='float32')
    r.write(HERE/'phase-schema.json',dict(names=names,phases=PHASES,target_mutation_checks=checks,core_definition='latest valid queued, no observed clear direct neighbor; neighbor coverage remains explicit',recovery_definition='historically queued but latest valid observation clear; not a future-clearance label'));np.savez_compressed(dest,**out);return out

def features(c,a,i):return np.column_stack([invariant_parent_features(c[f'{i}_Z']),a[f'{i}_A']])

def fit(months,regime):
    xs=[];ys=[];ws=[];ps=[];ds=[]
    for pop in months:
        meta,z,c=load(pop,'early');a=dict(np.load(HERE/f'phase-early-{pop}.npz'));counts=Counter(e['panel'] for e in meta)
        for i,e in enumerate(meta):
            ri=c[f'{i}_ri'];hi=c[f'{i}_hi'];m=z[f'{i}_m'][hi,ri]
            if not m.any():continue
            old=z[f'{i}_base'][hi,ri]>=.25;y=old!=z[f'{i}_y'][hi,ri].astype(bool);xs.append(features(c,a,i)[m]);ys.append(y[m]);ps.append(a[f'{i}_phase'][m]);ds.append(old[m]);ws.append(np.full(m.sum(),1/counts[e['panel']]/m.sum()))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);w/=w.mean();phase=np.concatenate(ps);direction=np.concatenate(ds);models={};stats=[]
    for label,take in [('shared',np.ones(len(x),bool))]+[(f'{p}-{d}',(phase==p)&(direction==bool(d))) for p in range(4) for d in range(2)]:
        if take.sum()<60 or len(np.unique(y[take]))<2:continue
        params=dict(objective='binary',num_leaves=7,learning_rate=.04,min_data_in_leaf=20,lambda_l2=15,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=31)
        model=lgb.train(params,lgb.Dataset(x[take],label=y[take],weight=w[take]),num_boost_round=160);model.save_model(str(HERE/f'phase-{regime}-{label}.txt'));models[label]=model;stats.append(dict(expert=label,rows=int(take.sum()),correctable=int(y[take].sum())))
    r.write(HERE/f'phase-{regime}-training.json',dict(months=months,experts=stats,features=x.shape[1],thresholds=dict(parent=.25,proposal=.5,verifier=.5),class_weight_boost=False,parent_confidence='absolute probabilities replaced by frozen binary/support features',teacher_fit='June-September; held out Oct/Nov'))
    return models

def evaluate(meta,z,c,a,backend,models,variant):
    cc=dict(c)
    class Identity:
        def predict(self,x,**kw):return x[:,0]
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        x=features(c,a,i);ri=c[f'{i}_ri'];hi=c[f'{i}_hi'];p=a[f'{i}_phase'];direction=z[f'{i}_base'][hi,ri]>=.25;accept=np.zeros(len(x),bool)
        if len(x):
            if variant=='shared':accept=models['shared'].predict(x,num_threads=4)>=.5
            else:
                for phase in range(4):
                    for d in range(2):
                        take=(p==phase)&(direction==bool(d));key=f'{phase}-{d}';model=models.get(key,models['shared'])
                        if take.any():accept[take]=model.predict(x[take],num_threads=4)>=.5
            if variant=='core_guard':accept[(p==0)|(p==3)]=False
            if variant=='boundary_only':accept[p!=1]=False
            if variant=='fresh_boundary':accept[(p!=1)|(a[f'{i}_A'][:,3]>0)]=False
        cc[f'{i}_Z']=accept[:,None].astype(float)
    return gate.metrics(meta,z,cc,backend,{'all':Identity()},'base' if variant=='base' else 'shared')

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args)
    for pop in ['october','november']:augment(backend,pop,'early')
    models=fit(['october'],'probe');meta,z,c=load('november','early');a=augment(backend,'november','early');variants=['base','shared','phase_experts','core_guard'];internal={k:evaluate(meta,z,c,a,backend,models,k)[0] for k in variants};b=internal['base'];eligible=[k for k in variants[1:] if internal[k]['ongoing_macro']>b['ongoing_macro'] and internal[k]['FP']<=b['FP'] and internal[k]['TP']>=b['TP']];selected=max(eligible,key=lambda k:internal[k]['ongoing_macro']) if eligible else 'base'
    r.write(args.output/'task2_queue_phase_internal.json',dict(scores=internal,selected=selected,later_month_selection=False,selection='November ongoing IoU improves, no increase in FP or FN cells',parent='R10 out-of-time surrogate'));r.write(HERE/'phase-frozen-policy.json',dict(selected=selected,thresholds=dict(parent=.25,proposal=.5,verifier=.5),later_month_selection=False));models=fit(['october','november'],'full');reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z,c=load(pop,'full');a=augment(backend,pop,'full');reports[pop]={}
        for k in variants:
            scores,rows=evaluate(meta,z,c,a,backend,models,k);reports[pop][k]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_queue_phase_external.json',dict(reports=reports,frozen_selection=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN','known_changed_cells']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
