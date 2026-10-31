"""Ongoing improvement with frozen best onset; Jan-only threshold calibration."""
import argparse,copy,gc,json,time,hashlib,shutil
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_onset_prior_context as prior
from onset_context_core import PastContext
from ongoing_core import features,observed_state
BASE=prior.BASE;OUT=BASE/'outputs/ongoing_20261030';RUN=BASE/'work/ongoing_20261030';SEEDS=prior.SEEDS
CONFIGS=[dict(name=f'{head}_{tag}',head=head,alpha=a) for head in ['direct','transition','hard_transition'] for tag,a in [('independent',1.),('blend25',.25)]]
def read(p):return prior.read(p)
def write(p,v):return prior.write(p,v)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)
def init():
    OUT.mkdir(exist_ok=True);RUN.mkdir(exist_ok=True)
    policy=dict(frozen_onset='early_onset_20261030/early_full, first3 threshold .50, last3 .55; same onset for both populations, no onset retraining or population-based switching',
      features='Reliable state/queue run/entries/exits, actual upstream/downstream queue fronts,108 causal fine features,8 previous-day ongoing risk priors,48 previous-day clock/link anomaly features,8 position bases,static and clock. T-masked mainline,T-5 ramps only.',
      supervision='All4039 positive ongoing +4104 no-known-positive June-Nov windows. Include every future-positive or recently observed queue link +<=64 random other supervised links. IPW corrects node sampling. Uniform2 known horizons/link; inverse probability corrects horizon sampling. Unknown cells excluded.',
      weights='Equal roads, positive-window/auxiliary-window80/20, natural known-cell mass within window. Historical priors describe observed support in this sampled roster, not true population incidence.',
      models='Direct shared-horizon binary classifier; separate queue-clearing and queue-entry heads conditioned on latest reliable queue state; hard variant gives2x loss to observed flips. Same520 rounds31leavesdepth6,minleaf100,L2=20,lr.04,maxbin127,8threads,three fixed seeds.',
      October='June-September weights and end-September context only. Six configurations: each new head alone or fixed25% mixture with R127 probabilities. At fixed.25 select top2 IoU among FP/FN<=1.10R127; fill with diagnostic IoU best if fewer than2. R127 itself fitted these dates, so R127-anchored development is not fully out of sample.',
      November='Top2,three seeds; report versus R127. No November model or threshold reselection. Final weights June-November.',
      January='Select only scalar threshold .10.. .55 step.025 maximizing IoU subject FP/FN<=1.05R127 and no road loss>.03. If no feasible threshold use diagnostic maxIoU. R127 threshold remains.25.',
      February='After models+January thresholds hash-frozen, evaluate both selected candidates on exactly same ongoing windows as R127. No February parameter changes; retain every IoU leader. Overall uses unchanged early_full onset.',
      evaluation='Reliable-observation proxy labels, not official S_queue. Reused February development populations, not a new blind test.',official_S_queue=None)
    write(OUT/'protocol.json',policy)
    root=BASE/'outputs/early_onset_20261030';files=[root/'January_results.json']+list((root/'shared').glob('final_*.txt'))+list(root.glob('final_early_*/model.txt'))
    write(OUT/'frozen_onset_hashes.json',{str(p.relative_to(BASE)):digest(p) for p in files})
def check_onset():
    for p,h in read(OUT/'frozen_onset_hashes.json').items():assert digest(BASE/p)==h
def setup():
    get,bases,rf,panels=prior.analog.setup();raw=prior.REF.entries('june-november-expanded-fit.json');daykeys={(e['panel'],e['origin'][:10]):e['day_key'] for e in raw}
    from task2q.data import day,network,make_sample
    nets={p:network(prior.REF.DATA,p) for p in panels}
    @lru_cache(maxsize=3)
    def arrays(panel,date):
        key=daykeys.get((panel,date));file=prior.REF.RUN/'rebuild/days'/f'{key}.npz'
        if key and file.exists():
            with np.load(file) as z:return {k:z[k] for k in z.files}
        return day(prior.REF.DATA,panel,date,cache_dir=RUN/'raw_day_cache')
    @lru_cache(maxsize=3)
    def indexed_day(key):
        path=prior.REF.RUN/'rebuild/days'/f'{key}.npz'
        if path.exists():
            with np.load(path) as z:return {k:z[k] for k in z.files}
        return None
    def sample(e):
        net=nets[e['panel']];values=indexed_day(e['day_key']);values=arrays(e['panel'],e['origin'][:10]) if values is None else values;s=make_sample(values,net,e['origin_slot'],e['panel']);assert s['condition']==1 and s['has_future_queue']==e['has_future_queue'];return s
    return sample,arrays,network,bases,rf,panels,raw
def prepare():
    init();get,arrays,network,bases,rf,panels,raw=setup();entries=[e for e in raw if e['condition']==1];assert len(entries)==8143;by_panel={p:[] for p in panels}
    for e in entries:by_panel[e['panel']].append(e)
    allmeta=[]
    for panel in panels:
        folder=RUN/'fit'/panel;folder.mkdir(exist_ok=True,parents=True)
        if (folder/'complete.json').exists():allmeta+=read(folder/'complete.json')['windows'];continue
        state=PastContext(len(bases[panel]['position']));net=network(prior.REF.DATA,panel);byday={}
        for e in by_panel[panel]:byday.setdefault(e['origin'][:10],[]).append(e)
        xx=[];yy=[];mm=[];ww=[];hh=[];months=[];metadata=[];cursor=0;rng=np.random.default_rng(20261101);checked=False;names=None
        for date in pd.date_range('2030-06-01','2030-11-30').strftime('%Y-%m-%d'):
            targets=[]
            for e in byday.get(date,[]):
                s=get(e);ramp=rf.ramp_history(s);counts=rf.topology(panel)['counts'];x,names,q=features(s,ramp,counts,bases[panel]['position'],state);y=s['y'].T;m=s['m'].T;targets.append((e['origin_slot'],y,m));known=m.any(1);_,_,_,g,qo=observed_state(s);forced=known&((y.astype(bool)&m).any(1)|qo[-6:].any(0));rest=np.flatnonzero(known&~forced);k=min(64,len(rest));pick=rng.choice(rest,k,replace=False) if k else rest;ids=np.r_[np.flatnonzero(forced),pick];pi=np.r_[np.ones(forced.sum()),np.full(k,k/max(len(rest),1))];h=np.full((len(ids),2),-1,dtype='int8')
                for j,node in enumerate(ids):
                    available=np.flatnonzero(m[node]);chosen=rng.choice(available,min(2,len(available)),replace=False);h[j,:len(chosen)]=chosen
                weight=(1/(max(int(m.sum()),1)*pi)).astype('float32');xx.append(x[ids]);yy.append(y[ids]);mm.append(m[ids]);ww.append(weight);hh.append(h);months.append(np.full(len(ids),int(date[:7].replace('-',''))));metadata.append(dict(panel=panel,origin=e['origin'],positive=bool(e['has_future_queue']),start=cursor,end=cursor+len(ids),selected_links=ids.tolist()));cursor+=len(ids)
                if not checked:
                    altered=copy.deepcopy(s);altered['y']=1-s['y'];altered['m']=~s['m'];altered['has_future_queue']=not s['has_future_queue'];z,nn,_=features(altered,ramp,counts,bases[panel]['position'],state);np.testing.assert_array_equal(x,z);assert names==nn;checked=True
            state.update_day(date,arrays(panel,date),rf.day(panel,'train',date),net,rf.topology(panel)['counts'],targets)
            if date in ['2030-09-30','2030-11-30']:
                out=OUT/('development_reference' if date=='2030-09-30' else 'final_reference');out.mkdir(exist_ok=True);state.save(out/f'{panel}.npz')
        for key,parts in [('X',xx),('y',yy),('m',mm),('w',ww),('H',hh),('month',months)]:np.save(folder/f'{key}.npy',np.concatenate(parts))
        write(folder/'complete.json',dict(rows=cursor,windows=metadata,feature_names=names,labels_not_features_checked=checked));allmeta+=metadata;write(OUT/'feature_names.json',dict(features=names,horizon_feature='horizon_minutes'));log(phase='FIT_PANEL',panel=panel,rows=cursor,features=len(names))
    write(RUN/'fit_complete.json',dict(windows=len(allmeta),panels=panels));prepare_query('development');log(phase='PREPARED')
def population_entries(pop):
    if pop=='development':return [e for e in read(BASE/'outputs/r127_calendar/monthly_event_roster.json')['samples'] if e['condition']==1 and e['origin'][:7] in ['2030-10','2030-11']]
    index={'january':'january-global-calibration.json','new_phase_population':'february-new_phase_population.json','old410':'february-old410.json'}[pop];return [e for e in prior.REF.entries(index) if e['condition']==1 and e['has_future_queue']]
def prepare_query(pop):
    if pop not in ['development','january']:assert (OUT/'frozen_before_February.json').exists()
    folder=RUN/pop;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    get,arrays,network,bases,rf,panels,raw=setup();entries=population_entries(pop);states={};xx=[];yy=[];mm=[];metadata=[];cursor=0;expected=read(OUT/'feature_names.json')['features']
    for e in entries:
        panel=e['panel'];s=get(e)
        if panel not in states:states[panel]=PastContext.load(OUT/('development_reference' if pop=='development' else 'final_reference')/f'{panel}.npz')
        x,n,q=features(s,rf.ramp_history(s),rf.topology(panel)['counts'],bases[panel]['position'],states[panel]);assert n==expected;xx.append(x);yy.append(s['y'].T);mm.append(s['m'].T);metadata.append(dict(panel=panel,origin=e['origin'],positive=True,start=cursor,end=cursor+len(x),entry=e));cursor+=len(x)
    for key,parts in [('X',xx),('y',yy),('m',mm)]:np.save(folder/f'{key}.npy',np.concatenate(parts))
    write(folder/'complete.json',dict(rows=cursor,windows=metadata));log(phase='QUERY_PREPARED',population=pop,windows=len(entries))
def load(pop):
    folder=RUN/pop;meta=read(folder/'complete.json');return {k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ['X','y','m','parent']},meta
def training(phase):
    chunks={k:[] for k in ['X','y','w','q']};names=read(OUT/'feature_names.json')['features'];qidx=names.index('current_observed_queue');metadata=[]
    for panel in read(RUN/'fit_complete.json')['panels']:
        folder=RUN/'fit'/panel;meta=read(folder/'complete.json');x=np.load(folder/'X.npy',mmap_mode='r');y=np.load(folder/'y.npy',mmap_mode='r');m=np.load(folder/'m.npy',mmap_mode='r');h=np.load(folder/'H.npy');w=np.load(folder/'w.npy').copy();use=np.load(folder/'month.npy')<203010 if phase=='development' else np.ones(len(x),bool);wins=[e for e in meta['windows'] if phase!='development' or e['origin'][:7]<'2030-10'];count={b:sum(e['positive']==b for e in wins) for b in [False,True]}
        for e in wins:w[e['start']:e['end']]*=(.8 if e['positive'] else .2)/count[e['positive']]
        ids,slot=np.nonzero((h>=0)&use[:,None]);hs=h[ids,slot];xs=np.column_stack([x[ids],(hs+1)*5]).astype('float32');known=m.sum(1);sampled=(h>=0).sum(1);ws=w[ids]*known[ids]/sampled[ids];chunks['X'].append(xs);chunks['y'].append(y[ids,hs].astype('float32'));chunks['w'].append(ws);chunks['q'].append(x[ids,qidx]>.5)
    out={k:np.concatenate(v) for k,v in chunks.items()};out['w']/=out['w'].mean();return out
def train(head,phase,seed):
    folder=OUT/f'{phase}_{head}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    data=training(phase);x=data['X'];w=data['w'];q=data['q'];branches=['direct'] if head=='direct' else ['entry','clear']
    for branch in branches:
        use=np.ones(len(x),bool) if branch=='direct' else q if branch=='clear' else ~q;y=1-data['y'][use] if branch=='clear' else data['y'][use];weight=w[use].copy()
        if head=='hard_transition':
            knownidx=read(OUT/'feature_names.json')['features'].index('current_state_known');observed_flip=(y>.5)&(x[use,knownidx]>.5);weight*=np.where(observed_flip,2.,1.)
        weight/=weight.mean()
        params=dict(objective='binary',learning_rate=.04,num_leaves=31,max_depth=6,min_data_in_leaf=100,lambda_l2=20,max_bin=127,feature_fraction=.8,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1)
        def cb(env):
            if (env.iteration+1)%100==0:log(phase=phase,head=head,branch=branch,seed=seed,round=env.iteration+1)
        model=lgb.train(params,lgb.Dataset(x[use],label=y,weight=weight),num_boost_round=520,callbacks=[cb]);model.save_model(str(folder/f'{branch}.txt'));np.save(folder/f'{branch}_gain.npy',model.feature_importance('gain'));del model;gc.collect()
    write(folder/'complete.json',dict(phase=phase,head=head,seed=seed,rows=len(x),dimensions=x.shape[1],branches=branches));del data,x,w,q;gc.collect()
def predict(head,phase,seed,data,pop):
    folder=OUT/f'{phase}_{head}_{seed}';file=folder/f'{pop}_probability.npy'
    if file.exists():return np.load(file)
    names=read(OUT/'feature_names.json')['features'];q=np.asarray(data['X'][:,names.index('current_observed_queue')])>.5;result=np.zeros((len(q),6),dtype='float32')
    for branch in (['direct'] if head=='direct' else ['entry','clear']):
        ids=np.arange(len(q)) if branch=='direct' else np.flatnonzero(q if branch=='clear' else ~q);model=lgb.Booster(model_file=str(folder/f'{branch}.txt'))
        for h in range(6):
            for start in range(0,len(ids),16384):
                ii=ids[start:start+16384];x=np.column_stack([data['X'][ii],np.full(len(ii),(h+1)*5)]);p=model.predict(x,num_threads=8);result[ii,h]=1-p if branch=='clear' else p
    assert np.isfinite(result).all();np.save(file,result);return result
def probability(cfg,phase,data,pop):
    p=np.mean([predict(cfg['head'],phase,s,data,pop) for s in SEEDS],0);return combine(cfg,p,data)
def combine(cfg,p,data):
    base=np.asarray(data['parent']);mixed=cfg['alpha']*p+(1-cfg['alpha'])*base;mode=cfg.get('mode')
    if mode:
        names=read(OUT/'feature_names.json')['features'];q=data['X'][:,names.index('current_observed_queue')]>.5
        if mode=='suppress_clear':mixed=np.where(q[:,None],np.minimum(base,mixed),base)
        elif mode=='confident_clear':mixed=np.where(q[:,None]&(p<.2),base*p,base)
        else:raise ValueError(mode)
    return mixed.astype('float32')
def score(p,d,m,threshold=.25):return prior.old.ar.score((p>=threshold).astype('float32'),d['y'],d['m'],m)
def compact(r):return prior.compact(r)
def subset(m,month):return prior.subset(m,month)
def diagnose(pop='development'):
    d,m=load(pop);names=read(OUT/'feature_names.json')['features'];x=d['X'];known=x[:,names.index('current_state_known')]>.5;q=x[:,names.index('current_observed_queue')]>.5;groups=np.where(~known,'unknown',np.where(q,'currently_queued','currently_clear'));rows=[]
    for w in m['windows']:
        a,b=w['start'],w['end']
        for group in ['currently_queued','currently_clear','unknown']:
            ids=np.flatnonzero(groups[a:b]==group)+a
            for h in range(6):
                y=d['y'][ids,h].astype(bool);mask=d['m'][ids,h];p=d['parent'][ids,h]>=.25;rows.append(dict(panel=w['panel'],origin=w['origin'],group=group,minutes=5*(h+1),TP=int((p&y&mask).sum()),FP=int((p&~y&mask).sum()),FN=int((~p&y&mask).sum()),TN=int((~p&~y&mask).sum())))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/f'{pop}_R127_error_groups.csv',index=False,encoding='utf-8-sig');summary=frame.groupby(['group','minutes'])[['TP','FP','FN','TN']].sum().reset_index();summary.to_csv(OUT/f'{pop}_R127_error_summary.csv',index=False,encoding='utf-8-sig');write(OUT/f'{pop}_R127_baseline.json',score(d['parent'],d,m));log(phase='DIAGNOSED',population=pop)
def screen():
    d,m=load('development');october=subset(m,'2030-10');base=score(d['parent'],d,october);results=[]
    for head in ['direct','transition','hard_transition']:
        train(head,'development',SEEDS[0]);p=predict(head,'development',SEEDS[0],d,'development')
        for cfg in [c for c in CONFIGS if c['head']==head]:
            mix=cfg['alpha']*p+(1-cfg['alpha'])*d['parent'];s=score(mix,d,october);eligible=s['cell_counts']['FP_cells']<=1.1*base['cell_counts']['FP_cells'] and s['cell_counts']['FN_cells']<=1.1*base['cell_counts']['FN_cells'];results.append(dict(config=cfg,eligible=bool(eligible),October=s));write(OUT/'October_search.json',results);log(phase='OCTOBER',config=cfg['name'],eligible=bool(eligible),**compact(s))
    top=sorted(results,key=lambda r:(r['eligible'],r['October']['onset_IoU']),reverse=True)[:2];write(OUT/'selected_candidates.json',top);log(phase='SCREEN_COMPLETE',candidates=[r['config']['name'] for r in top])
def confirm():
    d,m=load('development');nov=subset(m,'2030-11');base=score(d['parent'],d,nov);reports=[]
    for item in read(OUT/'selected_candidates.json'):
        cfg=item['config']
        for seed in SEEDS:train(cfg['head'],'development',seed)
        p=probability(cfg,'development',d,'development');s=score(p,d,nov);reports.append(dict(config=cfg,November=s,IoU_vs_R127=prior.REF.bootstrap(s,base)));log(phase='NOVEMBER',config=cfg['name'],**compact(s))
    write(OUT/'November_confirmation.json',dict(R127=base,candidates=reports))
def final_train():
    for item in read(OUT/'selected_candidates.json'):
        for seed in SEEDS:train(item['config']['head'],'final',seed)
    check_onset();prepare_query('january');log(phase='FINAL_COMPLETE')
def calibrate():
    check_onset();d,m=load('january');base=score(d['parent'],d,m);reports={}
    for item in read(OUT/'selected_candidates.json'):
        cfg=item['config'];p=probability(cfg,'final',d,'january');np.save(OUT/f"january_{cfg['name']}_probability.npy",p);grid=[];best=None;anybest=None
        for threshold in np.round(np.arange(.1,.5501,.025),3):
            s=score(p,d,m,float(threshold));c=s['cell_counts'];b=base['cell_counts'];eligible=c['FP_cells']<=1.05*b['FP_cells'] and c['FN_cells']<=1.05*b['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.03;record=dict(threshold=float(threshold),eligible=bool(eligible),**compact(s));grid.append(record)
            if anybest is None or s['onset_IoU']>anybest[1]['onset_IoU']:anybest=(record,s)
            if eligible and (best is None or s['onset_IoU']>best[1]['onset_IoU']):best=(record,s)
        selected=best or anybest;reports[cfg['name']]=dict(config=cfg,threshold=selected[0]['threshold'],feasible=best is not None,January=selected[1],grid=grid,IoU_vs_R127=prior.REF.bootstrap(selected[1],base));log(phase='JANUARY',config=cfg['name'],threshold=selected[0]['threshold'],feasible=best is not None,**compact(selected[1]))
    write(OUT/'January_results.json',dict(R127=base,candidates=reports));files=[OUT/'January_results.json',OUT/'protocol.json',OUT/'selected_candidates.json',OUT/'frozen_onset_hashes.json']+list(OUT.glob('*amendment*.json'))+list(OUT.glob('final_*/*.txt'))+list((OUT/'final_reference').glob('*.npz'));write(OUT/'frozen_before_February.json',{p.relative_to(OUT).as_posix():digest(p) for p in files});log(phase='FROZEN')
def prepare_february():
    for p,h in read(OUT/'frozen_before_February.json').items():assert digest(OUT/p)==h
    for pop in ['new_phase_population','old410']:prepare_query(pop)
def evaluate():
    for p,h in read(OUT/'frozen_before_February.json').items():assert digest(OUT/p)==h
    check_onset();cal=read(OUT/'January_results.json');onset=read(BASE/'outputs/early_onset_20261030/February_results.json');results={}
    for pop in ['new_phase_population','old410']:
        d,m=load(pop);base=score(d['parent'],d,m);out={}
        expected=read(BASE/'outputs/onset_experiment/r127_reproduction.json')['reports'][pop]['by_panel_condition'];assert abs(base['onset_IoU']-np.mean([v for k,v in expected.items() if k.endswith('/queue_ongoing')]))<1e-12
        for name,r in cal['candidates'].items():
            p=probability(r['config'],'final',d,pop);np.save(OUT/f'{pop}_{name}_probability.npy',p);s=score(p,d,m,r['threshold']);out[name]=dict(score=s,threshold=r['threshold'],IoU_vs_R127=prior.REF.bootstrap(s,base),fixed_onset_IoU=onset[pop]['candidates']['early_full']['score']['onset_IoU'],overall_IoU=(s['onset_IoU']+onset[pop]['candidates']['early_full']['score']['onset_IoU'])/2);log(phase='FEBRUARY',population=pop,config=name,overall=out[name]['overall_IoU'],**compact(s))
        results[pop]=dict(R127=base,candidates=out);write(OUT/'February_results.json',results)
    check_onset();log(phase='EVALUATION_COMPLETE')
if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','diagnose','screen','confirm','train-final','calibrate','prepare-february','evaluate']);a=ap.parse_args().action;{'prepare':prepare,'diagnose':diagnose,'screen':screen,'confirm':confirm,'train-final':final_train,'calibrate':calibrate,'prepare-february':prepare_february,'evaluate':evaluate}[a]()
