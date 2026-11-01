"""Matched-budget time/calendar/structural-region rescue experiment."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
import argparse,copy,gc,json,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import lightgbm as lgb
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/task2-gate'))
from gate import r,RESEARCH,HERE as GATE,predict_head
from context import Context

def progress(stage,**kw):
    d=dict(stage=stage,time=time.time(),**kw);r.write(HERE/'progress.json',d);print(json.dumps(d),flush=True)

def prepare(args,backend,ctx):
    entries=r.read(RESEARCH/'selected-training-windows.json');ctx.build_profiles(entries)
    done=HERE/'matrix.json'
    if done.exists():return
    n=len(np.load(RESEARCH/'matrix/y.npy',mmap_mode='r'));Y=np.load(RESEARCH/'matrix/y.npy',mmap_mode='r');M=np.load(RESEARCH/'matrix/m.npy',mmap_mode='r');P=np.load(RESEARCH/'matrix/panel.npy',mmap_mode='r')
    arrays={}
    for regime in ['early','full']:
        arrays[regime+'E']=np.lib.format.open_memmap(HERE/(regime+'-extra.npy'),mode='w+',dtype='float32',shape=(n,40))
        arrays[regime+'F']=np.lib.format.open_memmap(HERE/(regime+'-future.npy'),mode='w+',dtype='float32',shape=(n,6,10))
    early=np.zeros(n,bool);rng=np.random.default_rng(r.SEED);cursor=0;panel_codes={};checks=[]
    _,sample,features,_,_,_,unseen=backend
    for i,e in enumerate(entries):
        s=sample(e);y=s['y'].T.astype(bool);m=s['m'].T.astype(bool);eligible=unseen(s)&m.any(1);positive=(y&m).any(1);pos=np.flatnonzero(eligible&positive);neg=np.flatnonzero(eligible&~positive);kn=min(32,len(neg));ids=np.r_[pos,rng.choice(neg,kn,replace=False) if kn else np.array([],int)];end=cursor+len(ids)
        assert np.array_equal(y[ids],Y[cursor:end]) and np.array_equal(m[ids],M[cursor:end])
        code=int(P[cursor]);assert (P[cursor:end]==code).all();panel_codes[e['panel']]=code
        for regime in ['early','full']:
            ex,fe=ctx.extra(s,regime);arrays[regime+'E'][cursor:end]=ex[ids];arrays[regime+'F'][cursor:end]=fe[ids]
        early[cursor:end]=e['origin'][:10]<='2030-09-30'
        if e['panel'] not in [x['panel'] for x in checks]:
            ex,fe=ctx.extra(s,'full');changed=copy.deepcopy(s);changed['y']=1-changed['y'];changed['m']=~changed['m'];changed['has_future_queue']=not changed['has_future_queue'];ce,cf=ctx.extra(changed,'full');assert np.array_equal(ex,ce) and np.array_equal(fe,cf)
            v=copy.deepcopy(sample.load_day(e['panel'],e['origin'][:10]));slot=e['origin_slot']
            for key in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[key][slot+1:]=9999
            v['eligible'][slot+1:]=False;changed=sample.make_sample(v,s['network'],slot,e['panel']);ce,cf=ctx.extra(changed,'full');assert np.array_equal(ex,ce) and np.array_equal(fe,cf)
            checks.append(dict(panel=e['panel'],target_fields_invariant=True,query_future_invariant=True,current_day_excluded_from_profiles=True))
        cursor=end
        if (i+1)%128==0:progress('building_context_features',windows=i+1,total=len(entries))
    assert cursor==n
    for a in arrays.values():a.flush()
    np.save(HERE/'early.npy',early)
    r.write(done,dict(rows=n,base_features=916,extra_features=40,future_context_features=10,horizon_features=6,classifier_features=972,panel_codes=panel_codes,checks=checks))
    r.write(args.output/'task2_time_geo_causality.json',dict(passed=True,checks=checks,profile_training_day_excluded=True,holiday_flag_fabricated=False,real_city_label_fabricated=False))

def fit(args,regime):
    meta=r.read(HERE/'matrix.json');X=np.load(RESEARCH/'matrix/X.npy',mmap_mode='r');E=np.load(HERE/(regime+'-extra.npy'),mmap_mode='r');F=np.load(HERE/(regime+'-future.npy'),mmap_mode='r');Y=np.load(RESEARCH/'matrix/y.npy');M=np.load(RESEARCH/'matrix/m.npy');W=np.load(RESEARCH/'matrix/w.npy').astype(float);P=np.load(RESEARCH/'matrix/panel.npy');keep=np.load(HERE/'early.npy') if regime=='early' else np.ones(len(X),bool)
    for panel in ['shared',*meta['panel_codes']]:
        dest=HERE/(regime+'-'+panel+'.txt')
        if dest.exists():continue
        use=keep if panel=='shared' else keep&(P==meta['panel_codes'][panel]);ri,hi=np.nonzero(M&use[:,None]);xx=np.column_stack([X[ri],E[ri],np.eye(6,dtype='float32')[hi],F[ri,hi]]);yy=Y[ri,hi].astype(float);ww=W[ri]/M.sum(1)[ri];ww/=ww.mean()
        rounds=896 if panel=='shared' else 224;leaves=31 if panel=='shared' else 15
        params=dict(objective='binary',num_leaves=leaves,learning_rate=.05,min_data_in_leaf=40,lambda_l2=4,feature_fraction=.85,num_threads=4,verbosity=-1,seed=r.SEED,deterministic=True,force_col_wise=True,max_bin=63)
        def cb(env):
            if (env.iteration+1)%224==0:progress('training_context_experts',regime=regime,panel=panel,step=env.iteration+1,total=rounds)
        model=lgb.train(params,lgb.Dataset(xx,label=yy,weight=ww),num_boost_round=rounds,callbacks=[cb]);model.save_model(str(dest))
        r.write(dest.with_suffix('.json'),dict(regime=regime,panel=panel,link_rows=int(use.sum()),observed_cells=len(yy),features=xx.shape[1],rounds=model.current_iteration(),leaves=leaves))
        del xx,yy,ww,model;gc.collect()

def augmented_matrix(s,ctx,features,regime):
    x=features(s);e,f=ctx.extra(s,regime);n=len(x)
    return np.column_stack([np.tile(np.column_stack([x,e]),(6,1)),np.repeat(np.eye(6,dtype='float32'),n,axis=0),f.transpose(1,0,2).reshape(6*n,10)])

def score(meta,store,name,backend,routes=None,risk=None,cutoff=0,empty_parent=False):
    _,_,_,_,iou,aggregate,_=backend;records=[];neg=defaultdict(list);rows=[];addedTP=0;addedFP=0
    for i,e in enumerate(meta):
        b=store[f'{i}_base'];p=b.copy();kind=name
        if routes is not None:kind=routes.get(e['panel'],'base')
        if kind.endswith('_raw'):kind=kind[:-4]
        elif kind in ['shared','expert'] and risk is not None and risk[i]<cutoff:kind='base'
        if empty_parent and e['condition']==0 and (b>=.5).any():kind='base'
        if e['condition']==0 and kind!='base':
            q=store[f'{i}_{kind}'];use=store[f'{i}_unseen']&(b.max(0)<.5);p[:,use]=q[:,use]
        threshold=.5 if e['condition']==0 else .25;pred=p>=threshold;y=store[f'{i}_y'].astype(bool);m=store[f'{i}_m'];added=pred&~(b>=threshold)&m;addedTP+=int((added&y).sum());addedFP+=int((added&~y).sum())
        if not e['has_future_queue']:
            if e['condition']==0:neg[e['panel']].append(bool((pred&m).any()))
            continue
        v=iou(y,pred,m);row=dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],date=e['origin'][:10],iou=v);rows.append(row);records.append(row)
    result=aggregate(records);result['onset_macro']=float(np.mean([v for k,v in result['by_panel_condition'].items() if k.endswith('queue_onset')]))
    result['negative_by_road']={p:float(np.mean(v)) for p,v in neg.items()};result['negative_false_any']=float(np.mean(list(result['negative_by_road'].values()))) if neg else None;result['added_TP']=addedTP;result['added_FP']=addedFP
    return result,rows

def route(args,backend,ctx):
    parent,sample,features,_,_,_,unseen=backend;shared=lgb.Booster(model_file=str(HERE/'early-shared.txt'));old=lgb.Booster(model_file=str(GATE/'detector-through-2030-09.txt'));models={};reports={}
    for month in ['2030-10','2030-11']:
        dest=HERE/('route-'+month+'.npz');mp=dest.with_suffix('.json')
        if dest.exists():meta=r.read(mp);store=dict(np.load(dest))
        else:
            meta=r.read(GATE/('clean-'+month+'.json'));previous=np.load(GATE/('clean-'+month+'.npz'));store={}
            for i,e in enumerate(meta):
                ee=dict(e,origin_slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5);s=sample(ee);panel=e['panel']
                if panel not in models:models[panel]=lgb.Booster(model_file=str(HERE/('early-'+panel+'.txt')))
                xx=augmented_matrix(s,ctx,features,'early')
                for k,model in [('shared',shared),('expert',models[panel])]:store[f'{i}_{k}']=model.predict(xx,num_threads=4).reshape(6,-1).astype('float32')
                store[f'{i}_old']=predict_head(old,features(s))
                for k in ['base','y','m']:store[f'{i}_{k}']=previous[f'{i}_{k}']
                store[f'{i}_unseen']=unseen(s)
                if (i+1)%128==0:progress('selecting_context_roads',month=month,windows=i+1,total=len(meta))
            np.savez_compressed(dest,**store);r.write(mp,meta)
        risk=np.load(HERE/('route-'+month+'-risk.npy'));cutoff=r.read(HERE/'frozen-risk.json')['selected']['cutoff']
        reports[month]={k:score(meta,store,k,backend,risk=risk,cutoff=cutoff)[0] for k in ['base','old','shared_raw','expert_raw','shared','expert']}
    routes={};audit={}
    for panel in ctx.nets:
        candidates=[]
        for kind in ['shared','expert']:
            gains=[];negative_ok=True
            for month in reports:
                b=reports[month]['base'];c=reports[month][kind];key=panel+'/queue_onset';gains.append(c['by_panel_condition'][key]-b['by_panel_condition'][key]);negative_ok &= c['negative_by_road'].get(panel,0)<=b['negative_by_road'].get(panel,0)+1e-12
            if min(gains)>=0 and np.mean(gains)>=.004 and negative_ok:candidates.append((float(np.mean(gains)),kind))
            audit[panel+'/'+kind]=dict(monthly_gains=gains,negative_guard=bool(negative_ok),eligible=bool(min(gains)>=0 and np.mean(gains)>=.004 and negative_ok))
        routes[panel]=max(candidates)[1] if candidates else 'base'
    r.write(HERE/'frozen-routes.json',dict(routes=routes,audit=audit,selected_using='October/November early temporal surrogate only',threshold=.5,negative_guard='zero extra proxy-negative false-any in BOTH months',risk_cutoff=cutoff,risk_calibration='June-August fit, September calibration'))
    r.write(args.output/'task2_time_geo_early_results.json',dict(reports=reports,routes=routes,audit=audit,official_S_queue=None))
    print(json.dumps(dict(routes=routes,audit=audit)),flush=True)

def evaluate(args,backend,ctx):
    from risk import raw_risk_features
    _,sample,features,_,_,_,unseen=backend;frozen=r.read(HERE/'frozen-routes.json');assert frozen['negative_guard']=='zero extra proxy-negative false-any in BOTH months';routes=frozen['routes'];shared=lgb.Booster(model_file=str(HERE/'full-shared.txt'));models={};reports={};riskmodel=lgb.Booster(model_file=str(HERE/'risk-full.txt'));cutoff=frozen['risk_cutoff']
    populations=[('december','evaluation-december'),('january','evaluation-january'),('february_new','february-new_phase_population'),('february_old','february-old410')]
    for pop,stem in populations:
        dest=HERE/('eval-'+pop+'.npz');mp=dest.with_suffix('.json')
        if dest.exists():meta=r.read(mp);store=dict(np.load(dest))
        else:
            meta=r.read(GATE/(stem+'.json'));previous=np.load(GATE/(stem+'.npz'));store={}
            for i,e in enumerate(meta):
                for k in ['base','y','m']:store[f'{i}_{k}']=previous[f'{i}_{k}']
                if e['condition']==0:
                    ee=dict(e,origin_slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5);s=sample(ee);panel=e['panel']
                    if panel not in models:models[panel]=lgb.Booster(model_file=str(HERE/('full-'+panel+'.txt')))
                    xx=augmented_matrix(s,ctx,features,'full')
                    for k,model in [('shared',shared),('expert',models[panel])]:store[f'{i}_{k}']=model.predict(xx,num_threads=4).reshape(6,-1).astype('float32')
                    store[f'{i}_unseen']=unseen(s);store[f'{i}_old']=previous[f'{i}_q']
                if (i+1)%96==0:progress('testing_context_experts',population=pop,windows=i+1,total=len(meta))
            np.savez_compressed(dest,**store);r.write(mp,meta)
        riskpath=HERE/('eval-'+pop+'-risk.npy')
        if riskpath.exists():risk=np.load(riskpath)
        else:
            rr=[]
            for e in meta:
                if e['condition']==0:
                    ee=dict(e,origin_slot=int(e['origin'][11:13])*12+int(e['origin'][14:16])//5);rr.append(float(riskmodel.predict(raw_risk_features(sample(ee),ctx)[None],num_threads=4)[0]))
                else:rr.append(0.)
            risk=np.array(rr);np.save(riskpath,risk)
        scores={};rows={}
        for name in ['base','old','shared_raw','expert_raw','shared','expert','routed','routed_empty']:
            routed=name in ['routed','routed_empty'];scores[name],rows[name]=score(meta,store,'base' if routed else name,backend,routes if routed else None,risk,cutoff,empty_parent=name=='routed_empty')
        oldrows={(v['panel'],v['window_id']):v['iou'] for v in rows['base']}
        assert all(v['iou']>=oldrows[(v['panel'],v['window_id'])]-1e-12 for v in rows['routed_empty'])
        reports[pop]=dict(scores=scores,windows=rows)
        r.write(args.output/'task2_time_geo_external_results.json',dict(reports=reports,frozen_routes=routes,all_populations_reused_development=True,official_S_queue=None))
        print(json.dumps(dict(population=pop,scores={k:dict(onset=v['onset_macro'],overall=v['proxy_IoU'],negative=v['negative_false_any'],TP=v['added_TP'],FP=v['added_FP']) for k,v in scores.items()})),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','fit','route','evaluate']);ap.add_argument('--regime',choices=['early','full'])
    for k in ['package','source','data','output']:ap.add_argument('--'+k,type=Path,required=True)
    args=ap.parse_args();backend=r.setup(args);ctx=Context(args,backend,HERE/'profiles',r.read,r.write,progress)
    if args.action=='build':prepare(args,backend,ctx)
    elif args.action=='fit':fit(args,args.regime)
    elif args.action=='route':route(args,backend,ctx)
    else:evaluate(args,backend,ctx)
if __name__=='__main__':main()
