"""Bounded event-head search with previous-day context; frozen endpoint heads."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='8'
import argparse,copy,gc,json,hashlib,time,shutil
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_two_stage_interval as old
import run_analog_forecast as analog
from onset_context_core import PastContext
from onset_interval_core import compatible,event_grad,PATTERNS
BASE=old.BASE;REF=old.ref;RUN=BASE/'work/onset_prior_context_20261030';OUT=BASE/'outputs/onset_prior_context_20261030';SEEDS=old.SEEDS;SCREEN_SEED=SEEDS[0]
DEFAULT=dict(leaves=31,depth=6,min_leaf=100,l2=10,rounds=500)
def read(p):return old.read(p)
def write(p,v):return old.write(p,v)
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def names_for(variant):return ['prior'] if variant=='prior' else ['context'] if variant=='context' else ['prior','context'] if variant=='combined' else []
def protocol():
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    write(OUT/'protocol.json',dict(method='Stage1 historical site/clock prior plus current-state anomalies -> event activation; stage2 frozen previously trained start/end heads',
        no_hard_site_filter=True,context='Strictly earlier calendar dates for fitting rows. Mainline normal profiles are link x halfhour x weekday/weekend daily means, shrunk toward all weekdays; scales floored at [.03,.05,.02,.02,.02]. Targets only enter previous-day site/3h-clock priors, one exposure vote per day.',
        data_split='June-September weights and frozen end-September reference for October search / November confirmation; final June-November weights/reference; January thresholds only; February untouched by tuning',
        reference_policy='No same-day update before features; no Oct-Nov labels in development reference; no January labels/observations in final reference',
        search_features=['base940 (reuse)','prior','context','combined'],screen_seed=SCREEN_SEED,screen=DEFAULT,
        followup_hyperparameters=[dict(leaves=15,depth=4,min_leaf=160,l2=20,rounds=500),dict(leaves=63,depth=7,min_leaf=60,l2=20,rounds=600)],
        selection='Feature/hyperparameter choices ranked ONLY October at fixed .5, interval decoder. Top2 confirmed with3 seeds November. Require improvement>=.005, paired-day97.5%CI lower>0 (two comparisons), FP/FN<=1.05x baseline, no road loss>.03; pick highest October among passing. If none pass, highest October is diagnostic-only final candidate, cannot enter February.',
        January_gate='Only global threshold .10.. .95 step.05: maximize IoU subject FP<=R127,FN<=R127,no road loss>.03; adoption also November gate and paired-day95%CI lower>0. No feasible threshold => diagnostic best only.',
        caveat='All these public development populations have been inspected before; date-held-out weights do not make this a fresh blind test. Official S_queue unavailable.',official_S_queue=None))

def prepare():
    protocol();get,bases,rf,panels=analog.setup();fit=read(old.RUN/'fit_complete.json');n=fit['rows'];entries=REF.entries('june-november-expanded-fit.json');raw={(e['panel'],old.normtime(e['origin'])):e for e in entries};daykeys={(e['panel'],e['origin'][:10]):e['day_key'] for e in entries};fitmeta=read(BASE/'outputs/source_propagation/fit_metadata.json')
    from task2q.data import day,network
    @lru_cache(maxsize=3)
    def arrays(panel,date):
        dk=daykeys.get((panel,date));file=REF.RUN/'rebuild/days'/f'{dk}.npz'
        if dk and file.exists():
            with np.load(file) as z:return {k:z[k] for k in z.files}
        return day(REF.DATA,panel,date,cache_dir=RUN/'raw_day_cache')
    # Stable dimensions from the core API, not data labels.
    sample=fit['windows'][0];s=get(raw[(sample['panel'],old.normtime(sample['origin']))]);p,c,pn,cn=PastContext(s['x'].shape[1]).features(s,rf.ramp_history(s),rf.topology(s['panel'])['counts']);write(OUT/'feature_names.json',dict(prior=pn,context=cn,base=940))
    for key,d in [('prior',len(pn)),('context',len(cn))]:
        file=RUN/f'fit_{key}.npy'
        if not file.exists():a=np.lib.format.open_memmap(file,mode='w+',dtype='float32',shape=(n,d));a[:]=np.nan;a.flush()
    pp=np.lib.format.open_memmap(RUN/'fit_prior.npy',mode='r+');cc=np.lib.format.open_memmap(RUN/'fit_context.npy',mode='r+');checks=[]
    yy=np.load(REF.RUN/'fit/y.npy',mmap_mode='r');mm=np.load(REF.RUN/'fit/m.npy',mmap_mode='r')
    for panel in panels:
        finished=RUN/f'{panel}_complete.json'
        if finished.exists():checks+=read(finished)['checks'];continue
        net=network(REF.DATA,panel);assert list(net['link_ids'])==fitmeta[panel]['link_ids'];rnet=rf.topology(panel);state=PastContext(len(net['link_ids']));byday={}
        for w in fit['windows']:
            if w['panel']==panel:byday.setdefault(w['origin'][:10],[]).append(w)
        source=BASE/'work/source_propagation/fit'/panel;sy=np.load(source/'y.npy',mmap_mode='r');sm=np.load(source/'m.npy',mmap_mode='r');targetdays={}
        for i,origin in enumerate(fitmeta[panel]['origins']):
            t=pd.Timestamp(origin);targetdays.setdefault(origin[:10],[]).append((t.hour*12+t.minute//5,sy[i],sm[i]))
        panelchecks=[]
        for date in pd.date_range('2030-06-01','2030-11-30',freq='D').strftime('%Y-%m-%d'):
            reference=state.reference(date)
            for w in byday.get(date,[]):
                e=raw[(panel,old.normtime(w['origin']))];s=get(e);ramps=rf.ramp_history(s);p,c,npn,ncn=state.features(s,ramps,rnet['counts'],reference);assert npn==pn and ncn==cn;ids=np.asarray(w['selected_links']);a,b=w['start'],w['end'];pp[a:b]=p[ids];cc[a:b]=c[ids];np.testing.assert_array_equal(yy[a:b],s['y'].T[ids]);np.testing.assert_array_equal(mm[a:b],s['m'].T[ids])
                if not panelchecks:
                    altered=copy.deepcopy(s);altered['y']=1-s['y'];altered['m']=~s['m'];altered['has_future_queue']=not s['has_future_queue'];p2,c2,_,_=state.features(altered,ramps,rnet['counts'],reference);np.testing.assert_allclose(p,p2,equal_nan=True);np.testing.assert_allclose(c,c2,equal_nan=True);panelchecks.append(dict(panel=panel,labels_not_inputs=True,strictly_previous_day_reference=True))
            # Current-day raw states/targets become available ONLY for later dates.
            state.update_day(date,arrays(panel,date),rf.day(panel,'train',date),net,rnet['counts'],targetdays.get(date,[]))
            if date in ['2030-09-30','2030-11-30']:
                folder=OUT/('development_reference' if date=='2030-09-30' else 'final_reference');folder.mkdir(exist_ok=True);state.save(folder/f'{panel}.npz')
        pp.flush();cc.flush();checks+=panelchecks;write(finished,dict(passed=True,checks=panelchecks));log(phase='context_panel_complete',panel=panel)
    write(RUN/'fit_complete.json',dict(rows=n,prior_dimensions=len(pn),context_dimensions=len(cn),checks=checks,training_reference_cutoff='Previous day only'))
    if not (RUN/'development_complete.json').exists():
        data,meta=REF.load('fit') if False else (None,read(old.RUN/'development/complete.json'));pri=[];con=[]
        for w in meta['windows']:
            s=get(raw[(w['panel'],old.normtime(w['origin']))]);state=PastContext.load(OUT/'development_reference'/f"{w['panel']}.npz");p,c,_,_=state.features(s,rf.ramp_history(s),rf.topology(w['panel'])['counts']);pri.append(p);con.append(c)
        np.save(RUN/'development_prior.npy',np.concatenate(pri));np.save(RUN/'development_context.npy',np.concatenate(con));write(RUN/'development_complete.json',dict(rows=meta['rows'],windows=len(meta['windows']),reference_through='2030-09-30'))
    log(phase='PREPARE_COMPLETE',rows=n,prior=len(pn),context=len(cn))

def training(variant,phase):
    x,y,m,w=old.matrix('development' if phase=='development' else 'final','base940');use=np.load(old.RUN/'fit_month.npy')<203010 if phase=='development' else np.ones(len(y),bool)
    if variant!='base':x=np.column_stack([x]+[np.load(RUN/f'fit_{name}.npy',mmap_mode='r')[use] for name in names_for(variant)]).astype('float32')
    return x,y,m,w
def query(variant,pop='development'):
    if pop=='development':
        folder=old.RUN/'development';meta=read(folder/'complete.json');data={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ['X','y','m','parent']};prefix='development'
    else:
        data,meta=REF.load(pop);data['parent']=np.load(REF.RUN/pop/'parent.npy');prefix=pop
    x=np.asarray(data['X']);x=np.column_stack([x]+[np.load(RUN/f'{prefix}_{name}.npy') for name in names_for(variant)]) if variant!='base' else x
    return x,data,meta
def timing_file(phase,seed):
    file=OUT/'shared_timing'/f'{phase}_{seed}.txt'
    return file if file.exists() else old.OUT/f"{'development' if phase=='development' else 'final'}_base940_{seed}/timing.txt"
def freeze_timing():
    folder=OUT/'shared_timing';folder.mkdir(exist_ok=True);hashes={}
    for phase in ['development','final']:
        for seed in SEEDS:
            source=old.OUT/f'{phase}_base940_{seed}/timing.txt';target=folder/f'{phase}_{seed}.txt'
            if not target.exists():shutil.copy2(source,target)
            assert digest(source)==digest(target);hashes[target.relative_to(OUT).as_posix()]=digest(target)
    write(OUT/'shared_timing_hashes.json',hashes)
def train_one(config,seed,phase='development'):
    config_name=config['name'];variant=config['variant'];folder=OUT/f'{phase}_{config_name}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    x,y,m,w=training(variant,phase);pos=(y.astype(bool)&m).any(1);massfile=RUN/f'{phase}_compatibility_mass_{seed}.npy'
    if massfile.exists():mass=np.load(massfile)
    else:
        timing=lgb.Booster(model_file=str(timing_file(phase,seed)));mass=(old.endpoint_predict(timing,x[:,:940])*compatible(y,m)).sum(1);np.save(massfile,mass)
    def objective(z,data):return event_grad(z,mass,pos,w)
    params=dict(objective=objective,learning_rate=.04,num_leaves=config['leaves'],max_depth=config['depth'],min_data_in_leaf=config['min_leaf'],lambda_l2=config['l2'],feature_fraction=.8,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1,feature_pre_filter=False)
    def cb(env):
        if (env.iteration+1)%100==0:log(phase=phase,config=config_name,seed=seed,round=env.iteration+1)
    model=lgb.train(params,lgb.Dataset(x,label=pos.astype('float32')),num_boost_round=config['rounds'],callbacks=[cb]);model.save_model(str(folder/'event.txt'));np.save(folder/'event_gain.npy',model.feature_importance('gain'));write(folder/'complete.json',dict(trained=True,config=config,seed=seed,phase=phase,training_rows=len(x),frozen_timing_sha256=digest(timing_file(phase,seed))));del x,y,m,w,model;gc.collect()
def predict(config,seed,phase,x,pop):
    file=OUT/f"{phase}_{config['name']}_{seed}/{pop}_probability.npy"
    if file.exists():return np.load(file)
    model=lgb.Booster(model_file=str(OUT/f"{phase}_{config['name']}_{seed}/event.txt"));timing=lgb.Booster(model_file=str(timing_file(phase,seed)));q=old.endpoint_predict(timing,x[:,:940]);z=np.concatenate([model.predict(x[a:a+16384],raw_score=True,num_threads=8) for a in range(0,len(x),16384)]);e=np.exp(-np.logaddexp(0.,-z));p=(e[:,None]*(q@PATTERNS)).astype('float32');assert np.isfinite(p).all();np.save(file,p);return p
def subset(meta,month):return dict(meta,windows=[w for w in meta['windows'] if w['origin'].startswith(month)])
def compact(r):return dict(IoU=r['onset_IoU'],FP=r['cell_counts']['FP_cells'],FN=r['cell_counts']['FN_cells'],roads=r['per_road'])
def screen():
    configs=[dict(DEFAULT,name=v,variant=v) for v in ['prior','context','combined']];write(OUT/'initial_configs.json',configs);results=[]
    for config in configs:
        train_one(config,SCREEN_SEED);x,data,meta=query(config['variant']);p=predict(config,SCREEN_SEED,'development',x,'development');s=old.score(p,data,subset(meta,'2030-10'),.5,'interval');results.append(dict(config=config,October=compact(s)));write(OUT/'October_search.json',results);log(phase='OCTOBER_SCREEN',config=config['name'],**compact(s));del x;gc.collect()
    feature_choice=max(results,key=lambda r:r['October']['IoU'])['config']['variant'];write(OUT/'feature_search_choice.json',dict(feature_variant=feature_choice,selection='October only; one prespecified screening seed'))
    for name,params in [('regularized',dict(leaves=15,depth=4,min_leaf=160,l2=20,rounds=500)),('capacity',dict(leaves=63,depth=7,min_leaf=60,l2=20,rounds=600))]:
        config=dict(params,name=name,variant=feature_choice);train_one(config,SCREEN_SEED);x,data,meta=query(config['variant']);p=predict(config,SCREEN_SEED,'development',x,'development');s=old.score(p,data,subset(meta,'2030-10'),.5,'interval');results.append(dict(config=config,October=compact(s)));write(OUT/'October_search.json',results);log(phase='OCTOBER_HYPER',config=config['name'],**compact(s));del x;gc.collect()
    top=sorted(results,key=lambda r:r['October']['IoU'],reverse=True)[:2];write(OUT/'confirmation_candidates.json',top);log(phase='SEARCH_COMPLETE',top=[r['config']['name'] for r in top])

def confirm():
    candidates=read(OUT/'confirmation_candidates.json');reports=[];data,meta,x0=old.load_development('base940');baseline_p=np.mean([np.load(old.OUT/f'development_base940_{s}/development_probability.npy') for s in SEEDS],0);nov=subset(meta,'2030-11');baseline=old.score(baseline_p,data,nov,.5,'interval')
    for item in candidates:
        config=item['config'];ps=[]
        for seed in SEEDS:
            train_one(config,seed);x,d,m=query(config['variant']);ps.append(predict(config,seed,'development',x,'development'));del x;gc.collect()
        p=np.mean(ps,0);np.save(OUT/f"development_{config['name']}_ensemble.npy",p);s=old.score(p,data,nov,.5,'interval');ci=REF.bootstrap(s,baseline,reps=4000);# Expand to97.5% CI for two confirmatory candidates.
        aa=s['window_rows'];bb=baseline['window_rows'];days=sorted({r['origin'][:10] for r in aa});rng=np.random.default_rng(42);effects=[]
        for _ in range(4000):
            counts={d:int(v) for d,v in zip(*np.unique(rng.choice(days,len(days)),return_counts=True))};road={}
            for a,b in zip(aa,bb):
                k=counts.get(a['origin'][:10],0)
                if k:road.setdefault(a['panel'],[]).extend([a['iou']-b['iou']]*k)
            if len(road)==8:effects.append(np.mean([np.mean(v) for v in road.values()]))
        ci['day_cluster_97_5CI']=np.quantile(effects,[.0125,.9875]).tolist();eligible=s['onset_IoU']>=baseline['onset_IoU']+.005 and ci['day_cluster_97_5CI'][0]>0 and s['cell_counts']['FP_cells']<=baseline['cell_counts']['FP_cells']*1.05 and s['cell_counts']['FN_cells']<=baseline['cell_counts']['FN_cells']*1.05 and min(s['per_road'][r]-baseline['per_road'][r] for r in baseline['per_road'])>=-.03
        reports.append(dict(config=config,October=item['October'],November=s,vs_control=ci,passes=bool(eligible),seed_IoU=[old.score(z,data,nov,.5,'interval')['onset_IoU'] for z in ps]));log(phase='NOVEMBER_CONFIRM',config=config['name'],passed=bool(eligible),**compact(s))
    passed=[r for r in reports if r['passes']];chosen=max(passed or reports,key=lambda r:r['October']['IoU']);write(OUT/'November_confirmation.json',dict(control=baseline,candidates=reports));write(OUT/'selected_config.json',dict(config=chosen['config'],passes_November_gate=bool(passed),diagnostic_only_without_confirmation=not bool(passed),selected_by='Highest October score among November passing; otherwise highest October diagnostic candidate',November_candidate=chosen['November']));log(phase='CONFIRM_COMPLETE',config=chosen['config']['name'],gate=bool(passed))

def final_train():
    config=read(OUT/'selected_config.json')['config']
    for seed in SEEDS:train_one(config,seed,'final')
    log(phase='FINAL_TRAIN_COMPLETE')

def prepare_query(pop='january'):
    config=read(OUT/'selected_config.json')['config']
    for seed in SEEDS:assert (OUT/f"final_{config['name']}_{seed}/complete.json").exists()
    if pop!='january':assert read(OUT/'January_selection.json')['passes_stability_gate']
    index={'january':'january-global-calibration.json','new_phase_population':'february-new_phase_population.json','old410':'february-old410.json'}[pop];entries={(e['panel'],old.normtime(e['origin'])):e for e in REF.entries(index)};get,bases,rf,panels=analog.setup();data,meta=REF.load(pop);pri=[];ctx=[]
    for w in meta['windows']:
        s=get(entries[(w['panel'],old.normtime(w['origin']))]);state=PastContext.load(OUT/'final_reference'/f"{w['panel']}.npz");p,c,_,_=state.features(s,rf.ramp_history(s),rf.topology(w['panel'])['counts']);np.testing.assert_array_equal(s['y'].T,data['y'][w['start']:w['end']]);np.testing.assert_array_equal(s['m'].T,data['m'][w['start']:w['end']]);pri.append(p);ctx.append(c)
    np.save(RUN/f'{pop}_prior.npy',np.concatenate(pri));np.save(RUN/f'{pop}_context.npy',np.concatenate(ctx));write(RUN/f'{pop}_prepared.json',dict(reference_cutoff='2030-11-30',future_masks_not_features=True))

def calibrate():
    selected=read(OUT/'selected_config.json');config=selected['config'];x,data,meta=query(config['variant'],'january');ps=[predict(config,s,'final',x,'january') for s in SEEDS];p=np.mean(ps,0);np.save(OUT/'january_probability.npy',p);baseline=old.ar.score(data['parent'],data['y'],data['m'],meta);rows=[];scores=[]
    for threshold in np.round(np.arange(.1,.951,.05),3):
        s=old.score(p,data,meta,float(threshold),'interval');ok=s['cell_counts']['FP_cells']<=baseline['cell_counts']['FP_cells'] and s['cell_counts']['FN_cells']<=baseline['cell_counts']['FN_cells'] and min(s['per_road'][r]-baseline['per_road'][r] for r in baseline['per_road'])>=-.03;rows.append(dict(threshold=float(threshold),eligible=bool(ok),**compact(s)));scores.append(s)
    eligible=[i for i,r in enumerate(rows) if r['eligible']];best=max(eligible or list(range(len(rows))),key=lambda i:rows[i]['IoU']);candidate=scores[best];ci=REF.bootstrap(candidate,baseline);passed=bool(selected['passes_November_gate'] and eligible and ci['day_cluster_95CI'][0]>0)
    write(OUT/'January_selection.json',dict(config=config,threshold=rows[best]['threshold'],decoder='interval',grid=rows,passes_stability_gate=passed,passes_November_gate=selected['passes_November_gate'],feasible_threshold_exists=bool(eligible),IoU_vs_R127=ci));write(OUT/'January_results.json',dict(R127=baseline,candidate=candidate,seed_IoU=[old.score(z,data,meta,rows[best]['threshold'],'interval')['onset_IoU'] for z in ps]));files=[OUT/'January_selection.json']+[OUT/f"final_{config['name']}_{s}/event.txt" for s in SEEDS]+[timing_file('final',s) for s in SEEDS]+list((OUT/'final_reference').glob('*.npz'));write(OUT/'frozen_before_February.json',{p.resolve().relative_to(OUT.resolve()).as_posix():digest(p) for p in files});log(phase='JANUARY_CALIBRATED',gate=passed,**compact(candidate))

def evaluate():
    selected=read(OUT/'January_selection.json')
    if not selected['passes_stability_gate']:write(OUT/'February_not_evaluated.json',dict(reason='Predeclared November/January robustness gates failed; no new February predictions inspected'));log(phase='RETAIN_R127');return
    for p,h in read(OUT/'frozen_before_February.json').items():assert digest(OUT/p)==h
    reports={};config=selected['config']
    for pop in ['new_phase_population','old410']:
        prepare_query(pop);x,data,meta=query(config['variant'],pop);ps=[predict(config,s,'final',x,pop) for s in SEEDS];p=np.mean(ps,0);np.save(OUT/f'{pop}_probability.npy',p);r=old.ar.score(data['parent'],data['y'],data['m'],meta);s=old.score(p,data,meta,selected['threshold'],'interval');reports[pop]=dict(R127=r,candidate=s,IoU_vs_R127=REF.bootstrap(s,r));log(phase='FEBRUARY',population=pop,**compact(s))
    write(OUT/'February_results.json',reports)

if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','screen','confirm','train-final','prepare-january','calibrate','evaluate','freeze-timing']);args=ap.parse_args();{'prepare':prepare,'screen':screen,'confirm':confirm,'train-final':final_train,'prepare-january':prepare_query,'calibrate':calibrate,'evaluate':evaluate,'freeze-timing':freeze_timing}[args.action]()
