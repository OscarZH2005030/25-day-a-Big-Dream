"""Learn queue continuation/clearance near observed history fronts.

R127 threshold stays .25. New correction classifiers use fixed .5, and
continuous-speed corrections use the fixed published physical queue line.
"""
import argparse,json,warnings
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import lightgbm as lgb
from core import ROOT,r,Context,physical
from gate import gate_features
from joint import joint_features
from weak_features import weak_features

HERE=Path(__file__).resolve().parent/'ongoing';HERE.mkdir(exist_ok=True)
V2=ROOT/'work/task2-rescue-v2';CASCADE=HERE.parent

def queue_history(s):
    x=s['x'];n=x.shape[1];cut=s['network']['threshold']/s['network']['free_speed'];valid=(x[:,:,4]<.5)&(x[:,:,3]>=.75)&(x[:,:,7]>.5);q=valid&(x[:,:,0]<=cut)
    ever=q.any(0);last=np.zeros(n,bool);seen=np.zeros(n,bool);lastq=np.full(n,-1.);lastclear=np.full(n,-1.);transitions=np.zeros(n);run=np.zeros(n);longest=np.zeros(n)
    for t in range(13):
        transitions+=valid[t]&seen&(last!=q[t]);last=np.where(valid[t],q[t],last);seen|=valid[t];lastq[q[t]]=t;lastclear[valid[t]&~q[t]]=t
        run=np.where(q[t],run+1,0);longest=np.maximum(longest,run)
    down=np.zeros(n);up=np.zeros(n);front=ever.copy()
    for source,target in s['network']['edges'].T:
        down[source]+=last[target];up[target]+=last[source]
        if ever[source] or ever[target]:front[source]=True;front[target]=True
    latest=np.ones(n);old=np.ones(n)
    for t in range(13):
        latest[valid[t]]=x[t,valid[t],0]
        if t<=6:old[valid[t]]=x[t,valid[t],0]
    dynamic=front&((~last)|(transitions>0)|(latest>old))
    f=np.column_stack([ever,last,seen,q.sum(0)/13,valid.sum(0)/13,(12-lastq)/13,(12-lastclear)/13,transitions/12,longest/13,down/4,up/4,front,dynamic,latest-cut,latest-old]).astype('float32')
    return f,front,dynamic

def features(s,ctx,regime):
    n=s['x'].shape[1];zero=np.zeros((6,n));g=gate_features(s,zero,zero);extra,future=ctx.extra(s,regime);ph,_=physical(s)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);weak,_=weak_features(s)
    joint,_,_=joint_features(s);qh,zone,dynamic=queue_history(s);link=np.column_stack([g,extra,ph,qh]);a=np.concatenate([np.tile(link[:,None,:],(1,6,1)),weak,joint,future,np.tile(np.eye(6)[None],(n,1,1))],axis=2).astype('float32')
    assert np.isfinite(a).all();return a,zone,dynamic

def prepare(args,backend,ctx):
    dest=HERE/'train.npz'
    if dest.exists():return
    entries=r.read(args.source/'indices/june-november-expanded-fit.json')['samples'];by=defaultdict(list)
    for e in entries:
        if e['condition']==1:by[(e['panel'],e['origin'][:7])].append(e)
    rng=np.random.default_rng(20261029);selected=[]
    for key,group in sorted(by.items()):selected.extend([group[i] for i in sorted(rng.choice(len(group),min(32,len(group)),replace=False))])
    selected.sort(key=lambda e:(e['panel'],e['origin']));chunks=defaultdict(list);roster=[];cursor=0
    for i,e in enumerate(selected):
        s=backend[1](e);a,zone,dynamic=features(s,ctx,'full');ids=np.flatnonzero(zone);m=s['m'][:,ids].T;ri,hi=np.nonzero(m)
        if not len(ri):continue
        af,_,_=features(s,ctx,'early');xx=a[ids][ri,hi];earlyx=af[ids][ri,hi];y=s['y'][hi,ids[ri]];net=s['network'];raw=backend[1].load_day(e['panel'],e['origin'][:10]);speed=raw['speed'][e['origin_slot']+1:e['origin_slot']+7];target=(speed[hi,ids[ri]]-net['threshold'][ids[ri]])/net['free_speed'][ids[ri]]+xx[:,3]
        for k,v in dict(X=xx,earlyX=earlyx,y=y,target=target,w=np.full(len(ri),1/len(ri)),early=np.full(len(ri),e['origin'][:10]<='2030-09-30')).items():chunks[k].append(v)
        roster.append(dict(panel=e['panel'],origin=e['origin'],start=cursor,end=cursor+len(ri)));cursor+=len(ri)
        if (i+1)%128==0:print(json.dumps(dict(phase='ongoing_training_features',done=i+1,total=len(selected))),flush=True)
    np.savez_compressed(dest,**{k:np.concatenate(v) for k,v in chunks.items()});r.write(HERE/'train.json',dict(roster=roster,source_windows=len(selected),rows=cursor,selection='32 history-ongoing windows per road/month; all history queue zone links; no future-label sampling',weights='equal window per road/month; no label-class weighting'))

def fit(regime):
    z=np.load(HERE/'train.npz');take=z['early'] if regime=='early' else np.ones(len(z['y']),bool);x=z['earlyX' if regime=='early' else 'X'][take];w=z['w'][take];w=w/w.mean();out={}
    for objective,label in [('binary','y'),('regression','target')]:
        dest=HERE/f'{regime}-{objective}.txt'
        if dest.exists():out[objective]=lgb.Booster(model_file=str(dest));continue
        params=dict(objective=objective,num_leaves=15,learning_rate=.04,min_data_in_leaf=40,lambda_l2=10,feature_fraction=.9,num_threads=4,verbosity=-1,seed=20261029,deterministic=True,force_col_wise=True,max_bin=63)
        model=lgb.train(params,lgb.Dataset(x,label=z[label][take],weight=w),num_boost_round=256);model.save_model(str(dest));out[objective]=model
        print(json.dumps(dict(phase='ongoing_fit',regime=regime,objective=objective,rows=len(x))),flush=True)
    return out

def generate(args,backend,ctx,regime,pop):
    dest=HERE/f'{regime}-{pop}.npz'
    if dest.exists():return r.read(dest.with_suffix('.json')),dict(np.load(dest))
    models=fit(regime);store={}
    if regime=='early':
        allentries=r.read(args.source/'indices/june-november-expanded-fit.json')['samples'];month={'october':'2030-10','november':'2030-11'}[pop];meta=sorted([dict(e) for e in allentries if e['condition']==1 and e['origin'].startswith(month)],key=lambda e:(e['panel'],e['origin']));old=None
        from coverage_data import matrix
        parents={p:lgb.Booster(model_file=str(args.package/'payload/outputs/task2-round10-phase-coverage'/f'{p}-1-probe-expanded.txt')) for p in backend[3]}
    else:meta=r.read(CASCADE/f'full-{pop}.json');old=np.load(CASCADE/f'full-{pop}.npz')
    for i,e in enumerate(meta):
        if e['condition']!=1:continue
        slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5;s=backend[1](dict(e,origin_slot=slot));a,zone,dynamic=features(s,ctx,regime);ids=np.flatnonzero(zone);flat=a[ids].reshape(-1,a.shape[2]);base=parents[e['panel']].predict(matrix(s),num_threads=4).reshape(6,-1).astype('float32') if old is None else old[f'{i}_base'];binary=np.zeros_like(base);speed=np.ones_like(base)
        if len(flat):binary[:,ids]=models['binary'].predict(flat,num_threads=4).reshape(len(ids),6).T;speed[:,ids]=(models['regression'].predict(flat,num_threads=4)-flat[:,3]).reshape(len(ids),6).T
        for k,v in dict(base=base,y=s['y'],m=s['m'],binary=binary,speed=speed,zone=zone,dynamic=dynamic).items():store[f'{i}_{k}']=v
        if (i+1)%256==0:print(json.dumps(dict(phase='ongoing_predictions',pop=pop,done=i+1,total=len(meta))),flush=True)
    np.savez_compressed(dest,**store);r.write(dest.with_suffix('.json'),meta);return meta,store

def metrics(meta,z,backend,variant):
    rows=[];TP=FP=FN=0
    for i,e in enumerate(meta):
        if e['condition']!=1 or not e['has_future_queue']:continue
        b=z[f'{i}_base']>=.25;p=b.copy();y=z[f'{i}_y'].astype(bool);m=z[f'{i}_m']
        if variant!='base':
            mode,zone=variant.split('_');use=z[f'{i}_{zone}'];learned=(z[f'{i}_binary']>=.5) if mode=='binary' else (z[f'{i}_speed']<=0)
            p[:,use]=learned[:,use]
        rows.append(dict(panel=e['panel'],condition=1,window_id=e['origin'],date=e['origin'][:10],iou=backend[4](y,p,m)));TP+=int((p&y&m).sum());FP+=int((p&~y&m).sum());FN+=int((~p&y&m).sum())
    report=backend[5](rows);values=list(report['by_panel_condition'].values());report.update(ongoing_macro=float(np.mean(values)),TP=TP,FP=FP,FN=FN);return report,rows

def main():
    ap=argparse.ArgumentParser()
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);ctx=Context(args,backend,V2/'profiles',r.read,r.write,lambda *a,**kw:None);prepare(args,backend,ctx);fit('early');variants=['base','binary_zone','binary_dynamic','speed_zone','speed_dynamic'];internal={}
    for pop in ['october','november']:
        meta,z=generate(args,backend,ctx,'early',pop);internal[pop]={k:metrics(meta,z,backend,k)[0] for k in variants}
    eligible=[k for k in variants[1:] if all(v[k]['ongoing_macro']>v['base']['ongoing_macro'] and v[k]['FP']<=v['base']['FP'] and v[k]['TP']>=.995*v['base']['TP'] for v in internal.values())];selected=max(eligible,key=lambda k:np.mean([v[k]['ongoing_macro']-v['base']['ongoing_macro'] for v in internal.values()])) if eligible else 'base'
    r.write(args.output/'task2_ongoing_internal.json',dict(reports=internal,selected=selected,parent='June-September R10 ongoing probe surrogate, not exact R127',original_parent_threshold=.25,correction_classifier_threshold=.5,physical_speed_boundary=0,later_month_selection=False));r.write(HERE/'frozen-policy.json',dict(selected=selected,legacy_threshold=.25,new_classifier_threshold=.5,physical_speed_margin_boundary=0,later_month_selection=False));fit('full');reports={}
    for pop in ['december','january','february_new','february_old']:
        meta,z=generate(args,backend,ctx,'full',pop);reports[pop]={}
        for k in variants:
            scores,rows=metrics(meta,z,backend,k);reports[pop][k]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_ongoing_external.json',dict(reports=reports,selected_on_early_months=selected,all_evaluations_reused_development=True,official_truth_available=False));print(json.dumps(dict(population=pop,selected=selected,scores={k:{j:v['scores'][j] for j in ['ongoing_macro','TP','FP','FN']} for k,v in reports[pop].items()})),flush=True)

if __name__=='__main__':main()
