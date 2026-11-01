"""Bounded Task2 experiment; frozen R127 remains untouched. CPU, public proxies."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='4'
import argparse,copy,gc,hashlib,importlib,json,sys,time
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import lightgbm as lgb

HERE=Path(__file__).resolve().parent
SEED=20261028

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
def progress(phase,**kw):
    r=dict(phase=phase,time=time.time(),**kw);write(HERE/'progress.json',r);print(json.dumps(r),flush=True)

def setup(args):
    cfg=read(args.package/'profile.json');entry=args.package/'payload/outputs'/cfg['entry_round']
    sys.path.insert(0,str(entry));runtime=importlib.import_module(cfg['module'])
    model=getattr(runtime,cfg['class'])(entry)
    import ramp_features as rf
    rf.DATA=args.data.resolve();rf.WORK=HERE/'ramp-cache';rf.WORK.mkdir(exist_ok=True);rf.day.cache_clear();rf.topology.cache_clear()
    from task2q.data import network,day,make_sample,PANELS
    from task2q.metrics import window_iou,aggregate_metrics
    nets={p:network(args.data,p) for p in PANELS}
    @lru_cache(maxsize=2)
    def load_day(p,d):return day(args.data,p,d,cache_dir=HERE/'day-cache')
    def sample(e):
        s=make_sample(load_day(e['panel'],e['origin'][:10]),nets[e['panel']],int(e['origin_slot']),e['panel'])
        assert s['condition']==e['condition'] and s['has_future_queue']==e['has_future_queue']
        return s
    sample.load_day=load_day
    sample.make_sample=make_sample
    bases={}
    def features(s):
        p=s['panel']
        if p not in bases:
            import pandas as pd
            from position_mode_core import make_basis
            top=pd.read_csv(args.data/'corridors'/p/'network/lwr_mainline_topology.csv').set_index('mainline_link_id')
            order=top.reindex(nets[p]['link_ids']).order_index.to_numpy(dtype=float)
            bases[p],metadata=make_basis(order,np.isfinite(order),SEED)
            assert metadata['training_gate'],(p,metadata)
            for directory in [entry,entry.parent/'task2-round116-position-policy-preflight-repair']:
                saved=directory/(p+'-basis.npz')
                if saved.exists():
                    with np.load(saved) as f:assert np.allclose(f['position'],bases[p]['position'],atol=1e-12)
        return runtime.head_features(s,bases[p])
    return model,sample,features,nets,window_iou,aggregate_metrics,runtime.unseen_queue

def patterns(kind):
    if kind=='all64':return ((np.arange(64)[:,None]>>np.arange(6))&1).astype(float)
    a=[np.zeros(6)]
    for start in range(6):
        for end in range(start,6):
            v=np.zeros(6);v[start:end+1]=1;a.append(v)
    return np.array(a)

def compatibility(y,m,pats):return ((pats[None]==y[:,None])|~m[:,None]).all(2)
def softmax(raw):
    raw=raw-raw.max(1,keepdims=True);v=np.exp(raw);return v/v.sum(1,keepdims=True)
def partial_loss(raw,allowed,w=None):
    p=softmax(raw);post=softmax(np.where(allowed,raw,-np.inf));g=p-post;h=np.maximum(2*p*(1-p),1e-4)
    loss=-np.log(np.maximum((p*allowed).sum(1),1e-300))
    if w is not None:g*=w[:,None];h*=w[:,None];loss=np.average(loss,weights=w)
    else:loss=loss.mean()
    return float(loss),g,h

def check_math():
    rng=np.random.default_rng(SEED);raw=rng.normal(size=(4,64));y=rng.integers(2,size=(4,6));m=rng.random((4,6))>.4
    a=compatibility(y,m,patterns('all64'));loss,g,h=partial_loss(raw,a);eps=1e-5
    for i,j in [(0,0),(0,16),(1,63),(2,21),(3,7)]:
        hi=raw.copy();lo=raw.copy();hi[i,j]+=eps;lo[i,j]-=eps
        fd=(partial_loss(hi,a)[0]-partial_loss(lo,a)[0])/(2*eps)
        assert abs(fd-g[i,j]/len(raw))<1e-8
    assert np.allclose(partial_loss(raw,np.ones_like(a))[1],0)
    assert np.all(h>0)
    return dict(partial_gradient_finite_difference=True,unobserved_labels_zero_gradient=True)

def pick_entries(entries):
    rng=np.random.default_rng(SEED);groups=defaultdict(list)
    for e in entries:
        if e['condition']==0:groups[(e['panel'],e['origin'][:7],e['has_future_queue'])].append(e)
    chosen=[]
    for key,g in sorted(groups.items()):
        rng.shuffle(g);chosen.extend(g[:24 if key[2] else 8])
    return sorted(chosen,key=lambda e:(e['panel'],e['origin']))

def build(args,backend):
    _,sample,features,nets,_,_,unseen=backend
    done=HERE/'matrix/complete.json'
    if done.exists():return read(done)
    entries=pick_entries(read(args.source/'indices/june-november-expanded-fit.json')['samples']);write(HERE/'selected-training-windows.json',entries)
    counts=Counter((e['panel'],e['origin'][:7],e['has_future_queue']) for e in entries)
    rng=np.random.default_rng(SEED);chunks=defaultdict(list);audit=defaultdict(Counter);pattern_counts=Counter();checks=[]
    for i,e in enumerate(entries):
        s=sample(e);x=features(s);y=s['y'].T.astype(bool);m=s['m'].T.astype(bool);n=len(y);eligible=unseen(s)&m.any(1)
        p22=compatibility(y,m,patterns('interval22'));p64=compatibility(y,m,patterns('all64'));assert p64.any(1).all()
        complete=m.all(1);positive=(y&m).any(1);k=e['panel'];a=audit[k]
        a['windows']+=1;a['positive_windows']+=int(s['has_future_queue']);a['link_views']+=n
        a['complete_positive']+=int((complete&positive).sum());a['incompatible22_complete_positive']+=int((complete&positive&~p22.any(1)).sum())
        a['incompatible22_partial_positive']+=int((positive&~complete&~p22.any(1)).sum());a['partial_positive']+=int((positive&~complete).sum())
        a['history_missing_T_cells']+=int((s['x'][12,:,4]>.5).sum());a['history_total_T_cells']+=n
        code=(y*(1<<np.arange(6))).sum(1)
        pattern_counts.update(map(int,code[complete&positive]))
        # All observed positives; IPW random negatives, never silently impute a missing bit.
        pos=np.flatnonzero(eligible&positive);neg=np.flatnonzero(eligible&~positive);kn=min(32,len(neg));ids=np.r_[pos,rng.choice(neg,kn,replace=False) if kn else np.array([],int)]
        inv=np.r_[np.ones(len(pos)),np.full(kn,len(neg)/kn if kn else 0)]
        key=(e['panel'],e['origin'][:7],e['has_future_queue']);mass=(.8 if e['has_future_queue'] else .2)/counts[key]
        weight=inv/max(int(eligible.sum()),1)*mass
        for name,value in dict(X=x[ids],y=y[ids],m=m[ids],w=weight.astype('float32'),panel=np.full(len(ids),list(nets).index(k),dtype='int8')).items():chunks[name].append(value)
        if k not in [c['panel'] for c in checks]:
            q=copy.deepcopy(s);q['y']=1-q['y'];q['m']=~q['m'];q['has_future_queue']=not q['has_future_queue'];assert np.array_equal(features(s),features(q))
            v=copy.deepcopy(sample.load_day(k,e['origin'][:10]));slot=int(e['origin_slot'])
            for channel in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[channel][slot+1:]=9999
            v['eligible'][slot+1:]=False
            changed=sample.make_sample(v,nets[k],slot,k)
            assert np.array_equal(features(s),features(changed))
            checks.append(dict(panel=k,features_ignore_future_target_fields=True,future_mainline_perturbation_invariant=True))
        if (i+1)%32==0:progress('building_training_matrix',windows=i+1,total=len(entries))
    arrays={k:np.concatenate(v) for k,v in chunks.items()};folder=HERE/'matrix';folder.mkdir(exist_ok=True)
    for k,v in arrays.items():np.save(folder/(k+'.npy'),v)
    r=dict(rows=len(arrays['X']),features=arrays['X'].shape[1],windows=len(entries),audit=dict(audit),complete_positive_patterns=dict(pattern_counts),causality_checks=checks,fit_months='June-November',labels='noisy observation proxies',math_checks=check_math())
    write(done,r);write(args.output/'raw-data-pattern-audit.json',r);progress('matrix_complete',rows=r['rows'],windows=r['windows']);return r

def fit(args,kind):
    meta=read(HERE/'matrix/complete.json');X=np.load(HERE/'matrix/X.npy',mmap_mode='r');y=np.load(HERE/'matrix/y.npy');m=np.load(HERE/'matrix/m.npy');w=np.load(HERE/'matrix/w.npy').astype(float)
    if kind=='bce':
        rows,horizon=np.nonzero(m);xx=np.column_stack([X[rows],np.eye(6,dtype='float32')[horizon]])
        yy=y[rows,horizon].astype(float);ww=w[rows]/m.sum(1)[rows];ww/=ww.mean()
        params=dict(objective='binary',num_leaves=31,learning_rate=.05,min_data_in_leaf=40,lambda_l2=4,feature_fraction=.85,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True,max_bin=63)
        def cb(env):
            if (env.iteration+1)%128==0:progress('training',kind=kind,step=env.iteration+1,total=896)
        model=lgb.train(params,lgb.Dataset(xx,label=yy,weight=ww),num_boost_round=896,callbacks=[cb]);model.save_model(str(HERE/(kind+'.txt')))
        write(HERE/(kind+'-training.json'),dict(rows=len(xx),link_views=len(X),features=xx.shape[1],rounds=model.current_iteration(),trees=model.num_trees(),leaves=31))
        return model
    pats=patterns(kind);a=compatibility(y,m,pats);keep=a.any(1)&~a.all(1);w=w[keep];w/=w.mean();X=np.asarray(X[keep]);a=a[keep]
    rounds=64 if kind=='all64' else 186
    state={'loss':None}
    def objective(raw,ds):
        loss,g,h=partial_loss(raw,a,w);state['loss']=loss;return g.astype('float32'),h.astype('float32')
    params=dict(objective=objective,num_class=len(pats),num_leaves=7,learning_rate=.05,min_data_in_leaf=40,lambda_l2=4,feature_fraction=.85,num_threads=4,verbosity=-1,seed=SEED,deterministic=True,force_col_wise=True,max_bin=63)
    def callback(env):
        if (env.iteration+1)%16==0:
            env.model.save_model(str(HERE/(kind+'-checkpoint.txt')));progress('training',kind=kind,step=env.iteration+1,total=rounds,loss=state['loss'])
    model=lgb.train(params,lgb.Dataset(X,label=np.zeros(len(X)),free_raw_data=True),num_boost_round=rounds,callbacks=[callback]);model.save_model(str(HERE/(kind+'.txt')))
    write(HERE/(kind+'-training.json'),dict(rows=len(X),excluded_incompatible_or_unconstrained=int((~keep).sum()),features=X.shape[1],rounds=model.current_iteration(),trees=model.num_trees(),leaves=7,loss=state['loss'],pattern_count=len(pats),source_rows=meta['rows']))
    return model

def predictions(args,backend,population):
    model,sample,features,_,_,_,unseen=backend
    cache=HERE/('predictions-'+population+'.npz');meta_path=HERE/('predictions-'+population+'.json')
    cached_meta=read(meta_path) if cache.exists() else []
    if cached_meta and (population=='february' or any(not e['has_future_queue'] for e in cached_meta)):
        return cached_meta,dict(np.load(cache))
    entries=read(args.source/'indices'/({'december':'december-development.json','january':'january-global-calibration.json','february':'february-new_phase_population.json'}[population]))['samples']
    # The frozen scoring rosters contain positive windows only. Add a separately
    # reported sparse negative cohort rather than silently skipping the guard.
    if population in ['december','january']:
        month='2030-12' if population=='december' else '2031-01'
        from task2q.data import network
        for panel in ['D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S','D12_I5_N','D12_I5_S']:
            net=network(args.data,panel)
            for date in [month+'-'+str(d).zfill(2) for d in [4,12,20,28]]:
                values=sample.load_day(panel,date)
                for slot in [32,104,176,248]:
                    s=sample.make_sample(values,net,slot,panel)
                    if s['condition']==0 and not s['has_future_queue'] and s['m'].any():
                        entries.append(dict(panel=panel,origin=s['origin'],origin_slot=slot,condition=0,has_future_queue=False,cohort='fixed sparse natural-negative diagnostic'))
    write(HERE/(population+'-evaluation-roster.json'),entries)
    models={k:lgb.Booster(model_file=str(HERE/(k+'.txt'))) for k in ['all64','interval22','bce']}
    store=dict(np.load(cache)) if cached_meta else {};meta=cached_meta
    for i,e in enumerate(entries):
        if i<len(cached_meta):continue
        s=sample(e);b=model.probability(s);store[f'{i}_base']=b;store[f'{i}_y']=s['y'];store[f'{i}_m']=s['m'];store[f'{i}_unseen']=unseen(s)
        if s['condition']==0:
            x=features(s)
            for k,head in models.items():
                if k=='bce':
                    xx=np.column_stack([np.tile(x,(6,1)),np.repeat(np.eye(6,dtype='float32'),len(x),axis=0)])
                    store[f'{i}_{k}']=head.predict(xx,num_threads=4).reshape(6,-1).astype('float32')
                else:store[f'{i}_{k}']=(softmax(head.predict(x,raw_score=True,num_threads=4))@patterns(k)).T.astype('float32')
        meta.append(dict(panel=e['panel'],condition=e['condition'],origin=e['origin'],has_future_queue=e['has_future_queue']))
        if (i+1)%24==0:progress('predicting',population=population,windows=i+1,total=len(entries))
    np.savez_compressed(cache,**store);write(meta_path,meta);return meta,store

def score(backend,meta,store,kind='base',alpha=0.,threshold=.5):
    _,_,_,_,iou,aggregate,_=backend;records=[];rows=[];negative=[]
    for i,e in enumerate(meta):
        b=store[f'{i}_base'];p=b.copy();th=.25 if e['condition'] else threshold
        if e['condition']==0 and kind!='base':
            use=store[f'{i}_unseen'].copy()
            if kind in ['bce_rescue','bce_consensus']:
                use &= b.max(0)<.5;q=store[f'{i}_bce'];p[:,use]=q[:,use]
                if kind=='bce_consensus':
                    use &= store[f'{i}_interval22'].max(0)>=.5;p=b.copy();p[:,use]=q[:,use]
            else:
                q=store[f'{i}_{kind}'];p[:,use]=(1-alpha)*b[:,use]+alpha*q[:,use]
        y=store[f'{i}_y'];m=store[f'{i}_m'];pred=p>=th
        if not e['has_future_queue']:
            negative.append(dict(panel=e['panel'],condition=e['condition'],false_any=bool((pred&m).any()),false_fraction=float((pred&m).sum()/max(m.sum(),1))))
            continue
        v=iou(y,pred,m);record=dict(panel=e['panel'],condition=e['condition'],window_id=e['origin'],iou=v);records.append(record);rows.append(dict(**record,date=e['origin'][:10]))
    result=aggregate(records);result['onset_macro']=float(np.mean([v for k,v in result['by_panel_condition'].items() if k.endswith('queue_onset')]))
    if negative:
        groups=defaultdict(list)
        for r in negative:
            if r['condition']==0:groups[r['panel']].append(r)
        result['negative_onset_false_any_macro']=float(np.mean([np.mean([r['false_any'] for r in g]) for g in groups.values()])) if groups else None
    return result,rows

def evaluate(args,backend,population):
    meta,store=predictions(args,backend,population)
    if population=='december':
        candidates=[]
        # One predeclared threshold and three blend strengths; no road-specific search.
        for k in ['base','all64','interval22','bce']:
            for alpha in ([0.] if k=='base' else [.25,.5,1.]):
                report,rows=score(backend,meta,store,k,alpha);candidates.append(dict(kind=k,alpha=alpha,report=report));write(HERE/(f'december-{k}-{alpha}-windows.json'),rows)
        report,rows=score(backend,meta,store,'bce_rescue',1.);candidates.append(dict(kind='bce_rescue',alpha=1.,report=report));write(HERE/'december-bce_rescue-1.0-windows.json',rows)
        report,rows=score(backend,meta,store,'bce_consensus',1.);candidates.append(dict(kind='bce_consensus',alpha=1.,report=report));write(HERE/'december-bce_consensus-1.0-windows.json',rows)
        base=candidates[0]
        def negative_ok(c):
            a=c['report'].get('negative_onset_false_any_macro');b=base['report'].get('negative_onset_false_any_macro')
            return a is None or b is None or a<=b+.03
        eligible=[c for c in candidates if c['report']['onset_macro']>=base['report']['onset_macro']+.004 and negative_ok(c)]
        winner=max(eligible,key=lambda c:c['report']['onset_macro']) if eligible else base
        write(args.output/'december-experiment.json',dict(candidates=candidates,selected=winner,selection='onset macro +.004 over frozen R127, threshold .5; no per-road routing',official_S_queue=None))
        write(HERE/'frozen-selection.json',dict(kind=winner['kind'],alpha=winner['alpha'],threshold=.5,source='December reused development; rescue and consensus policies adaptively added after initial 10-candidate failure',candidate_count=12))
        print(json.dumps(dict(selected=winner['kind'],alpha=winner['alpha'],onset=winner['report']['onset_macro'],base_onset=base['report']['onset_macro'])),flush=True)
    else:
        selected=read(HERE/'frozen-selection.json');result,rows=score(backend,meta,store,**selected_filter(selected));base,brows=score(backend,meta,store)
        write(args.output/(population+'-frozen-comparison.json'),dict(selected=selected,baseline=base,candidate=result,windows=rows,baseline_windows=brows,official_S_queue=None,previously_inspected_development=True))

def selected_filter(s):return {k:s[k] for k in ['kind','alpha','threshold']}
def check_causality(args,backend):
    _,sample,features,nets,_,_,_=backend;entries=read(HERE/'selected-training-windows.json');checks=[]
    for panel in nets:
        e=next(e for e in entries if e['panel']==panel);s=sample(e);a=features(s)
        q=copy.deepcopy(s);q['y']=1-q['y'];q['m']=~q['m'];q['has_future_queue']=not q['has_future_queue'];assert np.array_equal(a,features(q))
        v=copy.deepcopy(sample.load_day(panel,e['origin'][:10]));slot=int(e['origin_slot'])
        for channel in ['speed','flow','occupancy','quality','origin_speed','origin_flow','origin_occupancy','origin_quality']:v[channel][slot+1:]=9999
        v['eligible'][slot+1:]=False;q=sample.make_sample(v,nets[panel],slot,panel);assert np.array_equal(a,features(q))
        checks.append(dict(panel=panel,target_perturbation_invariant=True,future_mainline_perturbation_invariant=True))
    write(args.output/'new-head-causality-checks.json',dict(passed=True,checks=checks,math=check_math()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['build','fit','check','december','january','february']);ap.add_argument('--kind',choices=['all64','interval22','bce']);ap.add_argument('--package',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(exist_ok=True)
    if args.action=='fit':fit(args,args.kind);return
    backend=setup(args)
    if args.action=='build':build(args,backend)
    elif args.action=='check':check_causality(args,backend)
    else:evaluate(args,backend,args.action)
if __name__=='__main__':main()
