"""Finalize fixed October candidates using actual new June-November origins."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,time,hashlib,gc,shutil,traceback
import numpy as np
import lightgbm as lgb
import train_ongoing_speed as speed
from speed_distribution_core import PANELS,odds_mix
B=speed.old.BASE;O=B/'outputs/dense_ongoing_final_v32_20261102';START=time.time()
PARTS=[B/'outputs/dense_ongoing_v28_20261102',B/'outputs/dense_ongoing_final_data_v30_20261102']
SEEDS=[20261030,20261031,20261101]
CONFIGS=[dict(name='dense_100',alpha=1.,role='primary'),dict(name='dense_50',alpha=.5,role='secondary')]
def save(n,x):speed.old.write(O/n,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(**x):print(json.dumps(dict(seconds=round(time.time()-START,1),**x)),flush=True)
def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();os.link(source,dest)
def frame(x,h):return np.column_stack([x,np.full((len(x),30),np.nan),np.full(len(x),h)]).astype('float32')
def parent_models(panel,seed,h):
    return lgb.Booster(model_file=str(O/f'base_models/final_{seed}/{panel}.txt')),lgb.Booster(model_file=str(O/f'prior_residuals/final_{seed}/{panel}-h{h}.txt'))

def training_data(panel):
    chunks={k:[] for k in ['X','y','m','w','month','positive','density']}
    for part in PARTS:
        root=part/'fit'/panel;meta=speed.old.read(root/'complete.json');x=np.load(root/'X.npy',mmap_mode='r');n=len(x)
        assert x.shape[1]==274
        for k in ['X','y','m','w','month']:chunks[k].append(np.load(root/f'{k}.npy',mmap_mode='r'))
        pos=np.zeros(n,bool);density=np.zeros(n)
        for win in meta['windows']:
            assert '2030-06'<=win['origin'][:7]<='2030-11'
            a,b=win['start'],win['end'];pos[a:b]=win['positive'];density[a:b]=win['density']
        chunks['positive'].append(pos);chunks['density'].append(density)
    d={k:np.concatenate(v) for k,v in chunks.items()};assert (d['month']>=203006).all() and (d['month']<=203011).all()
    return d

def train(seed,panel):
    d=training_data(panel)
    for h in range(5,31,5):
        i=h//5-1;ids=np.flatnonzero(d['m'][:,i]);tx=frame(d['X'][ids],h);oldx=np.column_stack([tx[:,:274],tx[:,-1]])
        base,prior=parent_models(panel,seed,h);assert base.num_feature()==prior.num_feature()==275
        initial=base.predict(oldx,raw_score=True,num_threads=4)+prior.predict(oldx,raw_score=True,num_threads=4)
        pos=d['positive'][ids];weight=d['w'][ids].astype(float)*np.where(pos,np.clip(.1/(d['density'][ids]+.01),.5,8.),1.)
        for label,mass in [(True,.8),(False,.2)]:
            use=pos==label;assert use.any();weight[use]*=mass/weight[use].sum()
        weight/=weight.mean()
        params=dict(objective='binary',learning_rate=.03,num_leaves=15,max_depth=4,min_data_in_leaf=100,lambda_l2=50,max_bin=127,feature_fraction=1.,bagging_fraction=.9,bagging_freq=1,num_threads=4,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1)
        model=lgb.train(params,lgb.Dataset(tx,label=d['y'][ids,i],weight=weight,init_score=initial),num_boost_round=100)
        p=O/f'models/final_{seed}/{panel}-h{h}.txt';p.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(p))
        log(stage='TRAIN_FINAL_DENSE',panel=panel,seed=seed,horizon=h,known_rows=len(ids));del model,tx,oldx,base,prior;gc.collect()
    del d;gc.collect()

def query(pop):
    assert pop in ['january','new_phase_population','old410']
    if pop!='january':assert (O/'frozen_before_February.json').exists()
    # Existing caches only: these calls must never prepare/write old data.
    for file in [speed.old.RUN/pop/'complete.json',speed.RUN/pop/'extra.npy']:assert file.exists(),str(file)
    d,meta=speed.query(pop);assert d['X'].shape[1]==274
    return d,meta

def probability(d,meta,alpha):
    out=np.empty(d['y'].shape,dtype='float32')
    for panel in PANELS:
        ids=np.concatenate([np.arange(w['start'],w['end']) for w in meta['windows'] if w['panel']==panel]);x=d['X'][ids];ps=[]
        for seed in SEEDS:
            z=np.empty((len(ids),6),dtype='float64')
            for i,h in enumerate(range(5,31,5)):
                base,prior=parent_models(panel,seed,h);xx=np.column_stack([x,np.full(len(x),h)])
                z[:,i]=base.predict(xx,raw_score=True,num_threads=4)+prior.predict(xx,raw_score=True,num_threads=4)
                if alpha:
                    model=lgb.Booster(model_file=str(O/f'models/final_{seed}/{panel}-h{h}.txt'));z[:,i]+=alpha*model.predict(frame(x,h),raw_score=True,num_threads=4)
            ps.append(np.exp(-np.logaddexp(0.,-z)))
        out[ids]=odds_mix(d['parent'][ids],np.mean(ps,axis=0).astype('float32'))
    return out

def main():
    O.mkdir(exist_ok=False)
    for part in PARTS:assert speed.old.read(part/'verification.json')['passed']
    selected=speed.old.read(B/'outputs/dense_ongoing_v29_20261102/selected_candidates.json')
    assert [r['config']['name'] for r in selected]==['dense_100','dense_50']
    save('decision_before_training.json',dict(configs_in_fixed_October_order=CONFIGS,fit='June-November actual-origin data, same v29 capacity/features/weights',data_parts=[str(p.relative_to(B)) for p in PARTS],origin_shifts_minutes=[-10,-5,5,10],correlated_adjacent_origins_not_independent_dates=True,teacher='Final frozen900+old100 logits; in-sample/nearby origin teacher, not OOF',model='100trees15leafdepth4min100L250lr.03 maxbin127 featurefraction1 bag.9; original3seeds; 305columns=274+30NaN+horizon',January='Only original smooth threshold family: intercept .05:.025:.75; slope 0,+/-.05,+/-.1,+/-.15; maxIoU subjectFP/FN<=1.05 R127 and perroadloss<=.03, otherwise diagnostic max',February='Both fixed candidates after full model/code/threshold freeze; same184/210ongoing windows; onset unchanged for each population; no reselection',production_onset_and_default_unchanged=True,Kaggle_upload=False))
    files=[Path(__file__),B/'outputs/dense_ongoing_final_runtime_v32_20261102.py',B/'outputs/current_task2_best.json',B/'outputs/ongoing_smooth_threshold_20261101/February_results.json',B/'outputs/ongoing_smooth_threshold_20261101/January_selection.json',B/'outputs/dense_ongoing_v29_20261102/selected_candidates.json',B/'outputs/dense_ongoing_v29_20261102/November_confirmation.json']
    for part in PARTS:
        files.extend([part/'verification.json',part/'statistics.json'])
        for panel in PANELS:files.extend(part/'fit'/panel/n for n in ['complete.json','X.npy','y.npy','m.npy','w.npy','month.npy'])
    for seed in SEEDS:
        for panel in PANELS:
            link(B/f'outputs/ongoing_jaccard_20261031/final_j075_{seed}/{panel}.txt',O/f'base_models/final_{seed}/{panel}.txt')
            for h in range(5,31,5):link(B/f'outputs/ongoing_horizon_20261031/final_bce_{seed}/{panel}-h{h}.txt',O/f'prior_residuals/final_{seed}/{panel}-h{h}.txt')
    for n in ['frozen_onset.zip','R127_frozen.zip']:link(B/'outputs/best_overall_horizon_20261031'/n,O/n)
    (O/'code').mkdir()
    for n in ['dense_ongoing_final_runtime_v32_20261102.py','ongoing_core.py','speed_distribution_core.py','onset_context_core.py']:
        shutil.copy2(B/'outputs'/n,O/'code'/n)
    files+=list((O/'base_models').rglob('*.txt'))+list((O/'prior_residuals').rglob('*.txt'))+list((O/'code').glob('*.py'))+[O/'frozen_onset.zip',O/'R127_frozen.zip']
    frozen_inputs={str(f.relative_to(B)):sha(f) for f in files};save('predeclared_hashes.json',frozen_inputs);log(stage='INPUTS_AND_PARENTS_RETAINED')
    for seed in SEEDS:
        for panel in PANELS:train(seed,panel)
    save('final_fit_complete.json',dict(passed=True,new_native_models=len(list((O/'models').rglob('*.txt'))),June_November_only=True))
    d,meta=query('january');base=probability(d,meta,0.);gold=np.load(B/'outputs/ongoing_horizon_20261031/january_bce_a100_probability.npy');err=float(np.max(abs(base-gold)));assert err<=1e-7
    save('January_incumbent_parity.json',dict(passed=True,error=err));baseline=speed.old.score(d['parent'],d,meta,.25);selections={}
    from dense_ongoing_final_runtime_v32_20261102 import Portable
    rt=Portable(O)
    for cfg in CONFIGS:
        prob=probability(d,meta,cfg['alpha']);np.save(O/f"january_{cfg['name']}_probability.npy",prob);checks=[]
        for panel in PANELS:
            win=next(w for w in meta['windows'] if w['panel']==panel);ids=np.arange(win['start'],min(win['end'],win['start']+32));out=rt.predict(panel,d['X'][ids],d['parent'][ids],cfg['alpha']);error=float(np.max(abs(out-prob[ids])));assert error<=1e-7;checks.append(dict(panel=panel,rows=len(ids),error=error))
        save(f"{cfg['name']}_January_runtime_check.json",dict(passed=True,input_only_no_labels=True,checks=checks));grid=[];best=None;unconstrained=None
        for a in np.round(np.arange(.05,.7501,.025),3):
            for b in [0.,-.05,.05,-.1,.1,-.15,.15]:
                t=np.clip(a+b*(np.arange(5,31,5)-17.5)/25,.05,.75);s=speed.old.score(prob,d,meta,t);c=s['cell_counts'];bc=baseline['cell_counts'];ok=c['FP_cells']<=1.05*bc['FP_cells'] and c['FN_cells']<=1.05*bc['FN_cells'] and min(s['per_road'][r]-baseline['per_road'][r] for r in baseline['per_road'])>=-.03
                row=dict(intercept=float(a),slope=b,thresholds=t.tolist(),IoU=s['onset_IoU'],FP=c['FP_cells'],FN=c['FN_cells'],eligible=bool(ok));grid.append(row)
                if unconstrained is None or s['onset_IoU']>unconstrained[1]['onset_IoU']:unconstrained=(row,s)
                if ok and (best is None or s['onset_IoU']>best[1]['onset_IoU']):best=(row,s)
        chosen=best or unconstrained;selections[cfg['name']]=dict(config=cfg,selected=chosen[0],January=chosen[1],feasible=best is not None,diagnostic_only=best is None);save(f"{cfg['name']}_January_grid.json",grid);log(stage='JANUARY_THRESHOLD_SELECTED',name=cfg['name'],threshold=chosen[0],feasible=best is not None)
    save('January_selection.json',dict(R127=baseline,candidates=selections));del rt,d;gc.collect()
    for f,h in frozen_inputs.items():assert sha(B/f)==h
    files=list((O/'models').rglob('*.txt'))+list((O/'base_models').rglob('*.txt'))+list((O/'prior_residuals').rglob('*.txt'))+list((O/'code').glob('*.py'))+[Path(__file__),O/'decision_before_training.json',O/'January_selection.json',O/'predeclared_hashes.json',O/'frozen_onset.zip',O/'R127_frozen.zip',B/'outputs/current_task2_best.json']
    frozen={str(f.relative_to(B)):sha(f) for f in files};save('frozen_before_February.json',frozen);log(stage='FULL_MODEL_AND_THRESHOLDS_FROZEN')
    incumbent=speed.old.read(B/'outputs/ongoing_smooth_threshold_20261101/February_results.json');default_threshold=np.asarray(speed.old.read(B/'outputs/ongoing_smooth_threshold_20261101/January_selection.json')['selected']['thresholds']);results={}
    for pop in ['new_phase_population','old410']:
        for f,h in frozen.items():assert sha(B/f)==h
        d,meta=query(pop);assert sum(w['positive'] for w in meta['windows'])=={'new_phase_population':184,'old410':210}[pop]
        bp=probability(d,meta,0.);gold=np.load(B/f'outputs/ongoing_horizon_20261031/{pop}_bce_a100_probability.npy');error=float(np.max(abs(bp-gold)));assert error<=1e-7
        base=speed.old.score(bp,d,meta,default_threshold);assert abs(base['onset_IoU']-incumbent[pop]['score']['onset_IoU'])<=1e-12
        r127=speed.old.score(d['parent'],d,meta,.25);reports={}
        for cfg in CONFIGS:
            row=selections[cfg['name']];prob=probability(d,meta,cfg['alpha']);np.save(O/f"{pop}_{cfg['name']}_probability.npy",prob);s=speed.old.score(prob,d,meta,np.asarray(row['selected']['thresholds']));fixed=incumbent[pop]['fixed_onset_IoU']
            reports[cfg['name']]=dict(config=cfg,thresholds=row['selected']['thresholds'],score=s,fixed_onset_IoU=fixed,overall_IoU=(fixed+s['onset_IoU'])/2,vs_default=speed.old.prior.REF.bootstrap(s,base),vs_R127=speed.old.prior.REF.bootstrap(s,r127),independent_raw_acceptance_pending=True)
            log(stage='FEBRUARY_REPORT',population=pop,name=cfg['name'],ongoing=s['onset_IoU'],overall=reports[cfg['name']]['overall_IoU'],FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'])
        results[pop]=dict(default=base,R127=r127,baseline_probability_error=error,candidates=reports);save('February_results.json',results);del d;gc.collect()
    for f,h in frozen.items():assert sha(B/f)==h
    save('verification.json',dict(passed=True,full_parent_chain_and_new_models_saved_before_February=True,June_November_only_fit=True,January_only_threshold=True,fixed_October_ranking_no_reselection=True,January_portable_parity=True,incumbent_probability_and_score_parity=True,production_onset_and_default_unchanged=True,independent_raw_acceptance_pending=True,official_score=None,Kaggle_upload=False));log(stage='COMPLETE')
if __name__=='__main__':
    try:main()
    except Exception as e:
        if O.exists():save('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),frozen_source_must_not_be_edited=True))
        raise
