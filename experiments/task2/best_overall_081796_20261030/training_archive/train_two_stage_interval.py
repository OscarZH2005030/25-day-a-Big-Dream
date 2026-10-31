"""Stable two-stage onset experiment: event support then partial-label endpoints.

Actions run in separate processes. No February data used in development or calibration.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='8'
import argparse,copy,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
import run_onset_experiment as ref
import run_anchored_residual as ar
import run_analog_forecast as analog
from onset_interval_core import *
BASE=ref.BASE;RUN=BASE/'work/two_stage_interval_20261030';OUT=BASE/'outputs/two_stage_interval_20261030'
SEEDS=[20261030,20261031,20261101];VARIANTS=['base940','fine'];EVENT_ROUNDS=500;TIME_ROUNDS=400
def read(p):return ref.read(p)
def write(p,v):return ref.write(p,v)
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)
def normtime(s):return s[:19]
def score(p,data,meta,threshold=.5,decoder='cells',positive_only=True):
    b=interval_decode(p,threshold) if decoder=='interval' else p>=threshold
    return ar.score(b.astype('float32'),data['y'],data['m'],meta,positive_only)
def prepare():
    RUN.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'protocol.json',dict(method='Two independent LightGBM heads: stage1 event support, stage2 endpoint-type-conditioned six-class start/end model; product endpoint distribution normalized over21 contiguous intervals',
        missing_labels='Endpoint allowed sets marginalized; stage1 zero-observed windows contribute likelihood (1-event)+event*compatible_interval_mass, never hard-labeled event negatives unless complete',
        exceptions='Known non-contiguous observations excluded from endpoint fitting, retained as event positives. No claim that all true events are contiguous.',
        fine_features='Reliable actual-time pair slopes/MAD/acceleration, demand-leading-deceleration consistency, directional neighbor gradients with age/coverage, and T-5 ramp demand changes',
        dates=dict(development_fit='June-September only',development_hold='Fixed hash sample128 positive +128 no-known-positive onset windows from October-November',final_fit='June-November',calibration='January only',February='Evaluate only if predeclared January robustness gate passes; never tune'),
        sampling='80/20 positive-window/auxiliary-window mass, equal roads. Inverse sampling correction within window; NO artificial .7 positive-link mass.',
        feature_choice='Oct-Nov fixed .5 threshold: fine selected only if ensemble IoU improves >=.005 and day-block 95% CI lower>0 and FP/FN <=1.05x base and no road loss>.03; otherwise retain base940',
        decoder_choice='Oct-Nov fixed .5: interval only if >=cell IoU and FP/FN<=1.05x; otherwise cells',
        seeds=SEEDS,event_rounds=EVENT_ROUNDS,time_rounds=TIME_ROUNDS,tree_params=dict(num_leaves=31,event_max_depth=6,timing_num_leaves=15,timing_max_depth=4,learning_rate=.04,lambda_l2=10,num_threads=8),
        January_gate='One threshold maximizing IoU subject FP<=R127,FN<=R127,no road IoU loss>.03; adoption additionally requires paired-day95% CI lower>0. If failed, do not inspect new February predictions.',official_S_queue=None))
    get,bases,rf,panels=analog.setup();raw={(e['panel'],normtime(e['origin'])):e for e in ref.entries('june-november-expanded-fit.json')}
    oldmeta=read(ref.RUN/'fit/complete.json');n=oldmeta['rows'];done=read(RUN/'fit_partial.json')['windows'] if (RUN/'fit_partial.json').exists() else 0
    first=oldmeta['windows'][0];sample=get(raw[(first['panel'],normtime(first['origin']))]);firstfine,names=fine_features(sample,rf.ramp_history(sample),rf.topology(sample['panel'])['counts']);d=len(names)
    mode='r+' if done else 'w+'
    z=np.lib.format.open_memmap(RUN/'fit_fine.npy',mode=mode,dtype='float32',shape=(n,d));w=np.lib.format.open_memmap(RUN/'fit_ipw.npy',mode=mode,dtype='float32',shape=(n,));months=np.lib.format.open_memmap(RUN/'fit_month.npy',mode=mode,dtype='int32',shape=(n,));records=[];checks=[]
    yy=np.load(ref.RUN/'fit/y.npy',mmap_mode='r');mm=np.load(ref.RUN/'fit/m.npy',mmap_mode='r')
    for j,e in enumerate(oldmeta['windows']):
        a,b=e['start'],e['end'];records.append(e)
        if j<done:continue
        s=get(raw[(e['panel'],normtime(e['origin']))]);f,nn=fine_features(s,rf.ramp_history(s),rf.topology(e['panel'])['counts']);assert nn==names;ids=np.asarray(e['selected_links']);np.testing.assert_array_equal(yy[a:b],s['y'].T[ids]);np.testing.assert_array_equal(mm[a:b],s['m'].T[ids])
        z[a:b]=f[ids];positive=(s['y'].T.astype(bool)&s['m'].T).any(1);known=s['m'].T.any(1);other=int((known&~positive).sum());sampled=int((~positive[ids]).sum());weights=np.where(positive[ids],1.,other/max(sampled,1));w[a:b]=weights/max(int(known.sum()),1);months[a:b]=int(e['origin'][:7].replace('-',''))
        if not any(c['panel']==e['panel'] for c in checks):
            altered=copy.deepcopy(s);altered['y']=1-s['y'];altered['m']=~s['m'];altered['has_future_queue']=not s['has_future_queue'];f2,_=fine_features(altered,rf.ramp_history(altered),rf.topology(e['panel'])['counts']);np.testing.assert_allclose(f,f2,equal_nan=True);checks.append(dict(panel=e['panel'],future_label_invariant=True))
        if (j+1)%100==0:
            for a0 in [z,w,months]:a0.flush()
            write(RUN/'fit_partial.json',dict(windows=j+1));log(phase='fine_features',windows=j+1,total=len(oldmeta['windows']),columns=d)
    for a0 in [z,w,months]:a0.flush()
    write(RUN/'fit_complete.json',dict(rows=n,windows=records,fine_names=names,causality_checks=checks));write(OUT/'feature_names.json',dict(base_dimensions=940,fine=names))
    del z,w,months
    # Full-link validation, independent of row subsampling. R127 has fitted these
    # months: its score is descriptive, never the rule for choosing fine features.
    roster=read(BASE/'outputs/r127_calendar/monthly_event_roster.json')['samples'];pos=[e for e in roster if e['condition']==0 and e['origin'][:7] in ['2030-10','2030-11']];assert len(pos)==128
    neg=[]
    for panel in panels:
        for month in ['2030-10','2030-11']:
            candidates=[e for e in raw.values() if e['panel']==panel and e['origin'].startswith(month) and e['condition']==0 and not e['has_future_queue']]
            assert len(candidates)>=8
            # Match clock distribution to positive origins, deterministic nearest
            # slot then hash; never inspect any model score for selection.
            for target in [e for e in pos if e['panel']==panel and e['origin'].startswith(month)]:
                chosen=min(candidates,key=lambda e:(abs(e['origin_slot']-target['origin_slot']),hashlib.sha256(e['origin'].encode()).hexdigest()));neg.append(chosen);candidates.remove(chosen)
    selected=sorted(pos+neg,key=lambda e:(e['panel'],e['origin']));assert len(selected)==256;write(OUT/'development_roster.json',dict(samples=selected))
    if not (RUN/'development/complete.json').exists():
        model,gets,r2,features,audit=ref.backend();xx=[];ff=[];yy=[];mm=[];pp=[];metadata=[];cursor=0
        for j,e in enumerate(selected):
            s=get(e);x=features(s);f,nn=fine_features(s,rf.ramp_history(s),rf.topology(s['panel'])['counts']);assert nn==names
            if e['has_future_queue']:
                path=BASE/'work/r127_calendar/event_predictions'/e['panel']/e['origin'][:7]/(e['origin'][:10]+f"-{e['origin_slot']}.npz")
                with np.load(path) as data:np.testing.assert_array_equal(s['y'],data['y']);np.testing.assert_array_equal(s['m'],data['mask']);p=data['probability'].T
            else:p=model.probability(s).T
            xx.append(x);ff.append(f);yy.append(s['y'].T);mm.append(s['m'].T);pp.append(p);count=len(x);metadata.append(dict(panel=e['panel'],origin=e['origin'],positive=bool(e['has_future_queue']),start=cursor,end=cursor+count));cursor+=count
            if (j+1)%32==0:log(phase='development_features',windows=j+1,total=len(selected))
        folder=RUN/'development';folder.mkdir(exist_ok=True)
        for key,value in [('X',xx),('fine',ff),('y',yy),('m',mm),('parent',pp)]:np.save(folder/f'{key}.npy',np.concatenate(value))
        write(folder/'complete.json',dict(rows=cursor,windows=metadata))
    log(phase='PREPARE_COMPLETE',rows=n,features=940+d)

def matrix(phase,variant):
    meta=read(RUN/'fit_complete.json');n=meta['rows'];x=np.load(ref.RUN/'fit/X.npy',mmap_mode='r')[:n];f=np.load(RUN/'fit_fine.npy',mmap_mode='r');months=np.load(RUN/'fit_month.npy');use=months<203010 if phase=='development' else np.ones(n,bool)
    # Subsetting only June-September is deliberate; no Oct-Nov row in dev loss.
    x=np.asarray(x[use]);x=np.column_stack([x,np.asarray(f[use])]) if variant=='fine' else x
    y=np.load(ref.RUN/'fit/y.npy',mmap_mode='r')[:n][use];m=np.load(ref.RUN/'fit/m.npy',mmap_mode='r')[:n][use];weight=np.load(RUN/'fit_ipw.npy').copy();counts={}
    for e in meta['windows']:
        if phase=='development' and e['origin'][:7]>='2030-10':continue
        key=(e['panel'],e['positive']);counts[key]=counts.get(key,0)+1
    for e in meta['windows']:
        if phase=='development' and e['origin'][:7]>='2030-10':continue
        weight[e['start']:e['end']]*=(.8 if e['positive'] else .2)/counts[(e['panel'],e['positive'])]
    weight=weight[use];weight/=weight.mean();return x,y,m,weight

def endpoint_predict(model,x):
    starts=[];ends=[]
    for a in range(0,len(x),16384):
        z=np.asarray(x[a:a+16384]);starts.append(softmax(model.predict(np.column_stack([z,np.zeros(len(z))]),num_threads=8,raw_score=True)));ends.append(softmax(model.predict(np.column_stack([z,np.ones(len(z))]),num_threads=8,raw_score=True)))
    return interval_distribution(np.concatenate(starts),np.concatenate(ends))

def predict(folder,x):
    timing=lgb.Booster(model_file=str(folder/'timing.txt'));event=lgb.Booster(model_file=str(folder/'event.txt'));q=endpoint_predict(timing,x)
    logits=np.concatenate([event.predict(np.asarray(x[a:a+16384]),num_threads=8,raw_score=True) for a in range(0,len(x),16384)]);p=np.exp(-np.logaddexp(0.,-logits))
    result=(p[:,None]*(q@PATTERNS)).astype('float32');assert np.isfinite(result).all() and (result>=0).all() and (result<=1+1e-6).all();return result

def train(phase):
    choice=read(OUT/'development_selection.json')['feature_variant'] if phase=='final' else None
    variants=[choice] if phase=='final' else VARIANTS
    for variant in variants:
        x,y,m,w=matrix(phase,variant);pos=(y.astype(bool)&m).any(1);allowed,representable=endpoint_allowed(y,m);ids=np.flatnonzero(pos&representable)
        tx=np.column_stack([np.repeat(x[ids],2,axis=0),np.tile([0.,1.],len(ids))]).astype('float32');aa=allowed[ids].reshape(-1,6);tw=np.repeat(w[ids],2);tw/=tw.mean()
        write(OUT/f'{phase}_{variant}_training_statistics.json',dict(rows=len(x),positive_sequences=int(pos.sum()),endpoint_sequences=len(ids),noncontiguous_observation_exceptions=int((pos&~representable).sum()),dimensions=x.shape[1]))
        for seed in SEEDS:
            folder=OUT/f'{phase}_{variant}_{seed}';folder.mkdir(exist_ok=True)
            if (folder/'complete.json').exists():continue
            common=dict(learning_rate=.04,lambda_l2=10,feature_fraction=.8,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1,feature_pre_filter=False)
            def tobj(preds,data):return endpoint_grad(preds,aa,tw)
            def callback(env):
                if (env.iteration+1)%100==0:log(phase=phase,variant=variant,seed=seed,head=HEAD,round=env.iteration+1)
            HEAD='timing'
            params=dict(common,objective=tobj,num_class=6,num_leaves=15,max_depth=4,min_data_in_leaf=40)
            timing=lgb.train(params,lgb.Dataset(tx,label=np.zeros(len(tx))),num_boost_round=TIME_ROUNDS,callbacks=[callback]);timing.save_model(str(folder/'timing.txt'))
            q=endpoint_predict(timing,x);comp=compatible(y,m);mass=(q*comp).sum(1)
            def eobj(preds,data):return event_grad(preds,mass,pos,w)
            HEAD='event';event=lgb.train(dict(common,objective=eobj,num_leaves=31,max_depth=6,min_data_in_leaf=100),lgb.Dataset(x,label=pos.astype('float32')),num_boost_round=EVENT_ROUNDS,callbacks=[callback]);event.save_model(str(folder/'event.txt'))
            write(folder/'complete.json',dict(trained=True,phase=phase,variant=variant,seed=seed,event_rounds=EVENT_ROUNDS,time_rounds=TIME_ROUNDS,train_rows=len(x),time_rows=len(tx),threshold_not_selected=True))
            # Gain is model reliance, not causal importance.
            np.savez(folder/'feature_gain.npz',event=event.feature_importance('gain'),timing=timing.feature_importance('gain'))
            log(phase='model_complete',training_phase=phase,variant=variant,seed=seed)
            del event,timing,q,comp,mass;gc.collect()
        del x,y,m,w,tx;gc.collect()
    log(phase=phase+'_TRAIN_COMPLETE')

def load_development(variant):
    folder=RUN/'development';meta=read(folder/'complete.json');data={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ['X','fine','y','m','parent']};x=np.asarray(data['X']);x=np.column_stack([x,data['fine']]) if variant=='fine' else x;return data,meta,x

def select():
    results={};probs={}
    for variant in VARIANTS:
        data,meta,x=load_development(variant);ps=[]
        for seed in SEEDS:
            folder=OUT/f'development_{variant}_{seed}';assert (folder/'complete.json').exists();p=predict(folder,x);ps.append(p);np.save(folder/'development_probability.npy',p)
        p=np.mean(ps,0);probs[variant]=p;np.save(OUT/f'development_{variant}_ensemble.npy',p)
        results[variant]={d:score(p,data,meta,.5,d) for d in ['cells','interval']}
        results[variant]['seed_cell_IoU']=[score(z,data,meta)['onset_IoU'] for z in ps]
        results[variant]['all_windows_cells']=score(p,data,meta,.5,'cells',False)
    base=results['base940']['cells'];fine=results['fine']['cells'];ci=ref.bootstrap(fine,base);eligible=fine['onset_IoU']>=base['onset_IoU']+.005 and ci['day_cluster_95CI'][0]>0 and fine['cell_counts']['FP_cells']<=1.05*base['cell_counts']['FP_cells'] and fine['cell_counts']['FN_cells']<=1.05*base['cell_counts']['FN_cells'] and min(fine['per_road'][p]-base['per_road'][p] for p in base['per_road'])>=-.03
    variant='fine' if eligible else 'base940';cells=results[variant]['cells'];interval=results[variant]['interval'];decoder='interval' if interval['onset_IoU']>=cells['onset_IoU'] and interval['cell_counts']['FP_cells']<=cells['cell_counts']['FP_cells']*1.05 and interval['cell_counts']['FN_cells']<=cells['cell_counts']['FN_cells']*1.05 else 'cells'
    write(OUT/'development_results.json',results);write(OUT/'development_selection.json',dict(feature_variant=variant,decoder=decoder,fine_passed_robustness_gate=bool(eligible),fine_vs_base=ci,threshold_fixed_at=.5,selection_dates='October-November only',R127_not_used_for_feature_selection=True))
    log(phase='DEVELOPMENT_SELECTED',feature=variant,decoder=decoder,base_IoU=base['onset_IoU'],fine_IoU=fine['onset_IoU'],CI=ci)

def prepare_queries():
    choice=read(OUT/'development_selection.json');variant=choice['feature_variant']
    for seed in SEEDS:assert (OUT/f'final_{variant}_{seed}/complete.json').exists()
    # January only. February preparation is allowed only after calibration gate.
    get,bases,rf,panels=analog.setup();entries={ (e['panel'],normtime(e['origin'])):e for e in ref.entries('january-global-calibration.json')}
    data,meta=ref.load('january');extras=[]
    for w in meta['windows']:
        s=get(entries[(w['panel'],normtime(w['origin']))]);f,names=fine_features(s,rf.ramp_history(s),rf.topology(s['panel'])['counts']);assert names==read(OUT/'feature_names.json')['fine'];np.testing.assert_array_equal(s['y'].T,data['y'][w['start']:w['end']]);np.testing.assert_array_equal(s['m'].T,data['m'][w['start']:w['end']]);extras.append(f)
    np.save(RUN/'january_fine.npy',np.concatenate(extras));write(RUN/'january_prepared.json',dict(after_all_final_models=True,windows=len(meta['windows'])));log(phase='JANUARY_PREPARED')

def calibrate():
    choice=read(OUT/'development_selection.json');variant=choice['feature_variant'];decoder=choice['decoder'];data,meta=ref.load('january');x=np.asarray(data['X']);x=np.column_stack([x,np.load(RUN/'january_fine.npy')]) if variant=='fine' else x
    ps=[predict(OUT/f'final_{variant}_{seed}',x) for seed in SEEDS];p=np.mean(ps,0);np.save(OUT/'january_probability.npy',p);base=ar.score(np.load(ref.RUN/'january/parent.npy'),data['y'],data['m'],meta)
    grid=np.round(np.arange(.10,.951,.05),3).tolist();rows=[];scores=[]
    for t in grid:
        s=score(p,data,meta,t,decoder);eligible=s['cell_counts']['FP_cells']<=base['cell_counts']['FP_cells'] and s['cell_counts']['FN_cells']<=base['cell_counts']['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.03
        rows.append(dict(threshold=t,IoU=s['onset_IoU'],FP=s['cell_counts']['FP_cells'],FN=s['cell_counts']['FN_cells'],eligible=bool(eligible)));scores.append(s)
    available=[i for i,r in enumerate(rows) if r['eligible']];best=max(available,key=lambda i:rows[i]['IoU']) if available else int(np.argmax([r['IoU'] for r in rows]));s=scores[best];ci=ref.bootstrap(s,base);passed=bool(available and ci['day_cluster_95CI'][0]>0)
    write(OUT/'January_selection.json',dict(threshold=grid[best],decoder=decoder,feature_variant=variant,grid=rows,feasible_threshold_exists=bool(available),passes_stability_gate=passed,selected_candidate_is_diagnostic_if_gate_failed=not passed,IoU_vs_R127=ci))
    write(OUT/'January_results.json',dict(R127=base,candidate=s,individual_seed_IoU=[score(z,data,meta,grid[best],decoder)['onset_IoU'] for z in ps],all_onset=score(p,data,meta,grid[best],decoder,False)))
    hashes={file.relative_to(OUT).as_posix():hashlib.sha256(file.read_bytes()).hexdigest() for seed in SEEDS for file in (OUT/f'final_{variant}_{seed}').glob('*.txt')};hashes['January_selection.json']=hashlib.sha256((OUT/'January_selection.json').read_bytes()).hexdigest();write(OUT/'frozen_before_February.json',hashes)
    log(phase='JANUARY_CALIBRATED',candidate=s['onset_IoU'],R127=base['onset_IoU'],threshold=grid[best],gate=passed)

def evaluate():
    selected=read(OUT/'January_selection.json')
    if not selected['passes_stability_gate']:
        write(OUT/'February_not_evaluated.json',dict(reason='January predeclared stability gate failed; retain R127; no new February candidate prediction or score inspected',official_S_queue=None));log(phase='RETAIN_R127_NO_FEBRUARY');return
    for file,h in read(OUT/'frozen_before_February.json').items():assert hashlib.sha256((OUT/file).read_bytes()).hexdigest()==h
    get,bases,rf,panels=analog.setup();reports={};variant=selected['feature_variant']
    for pop,file in [('new_phase_population','february-new_phase_population.json'),('old410','february-old410.json')]:
        data,meta=ref.load(pop);x=np.asarray(data['X']);entries={(e['panel'],normtime(e['origin'])):e for e in ref.entries(file)}
        if variant=='fine':
            ff=[]
            for w in meta['windows']:
                s=get(entries[(w['panel'],normtime(w['origin']))]);f,names=fine_features(s,rf.ramp_history(s),rf.topology(s['panel'])['counts']);assert names==read(OUT/'feature_names.json')['fine'];np.testing.assert_array_equal(s['y'].T,data['y'][w['start']:w['end']]);np.testing.assert_array_equal(s['m'].T,data['m'][w['start']:w['end']]);ff.append(f)
            x=np.column_stack([x,np.concatenate(ff)])
        ps=[predict(OUT/f'final_{variant}_{seed}',x) for seed in SEEDS];p=np.mean(ps,0);np.save(OUT/f'{pop}_probability.npy',p);baseline=ar.score(np.load(ref.RUN/pop/'parent.npy'),data['y'],data['m'],meta);s=score(p,data,meta,selected['threshold'],selected['decoder']);reports[pop]=dict(R127=baseline,candidate=s,IoU_vs_R127=ref.bootstrap(s,baseline),individual_seed_IoU=[score(z,data,meta,selected['threshold'],selected['decoder'])['onset_IoU'] for z in ps]);log(phase='FEBRUARY_EVALUATED',population=pop,IoU=s['onset_IoU'],R127=baseline['onset_IoU'])
    write(OUT/'February_results.json',reports)

if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train-dev','select','train-final','prepare-january','calibrate','evaluate']);args=ap.parse_args()
    {'prepare':prepare,'train-dev':lambda:train('development'),'select':select,'train-final':lambda:train('final'),'prepare-january':prepare_queries,'calibrate':calibrate,'evaluate':evaluate}[args.action]()
