"""Early-onset timing experiments. No February access until January freeze."""
import argparse,copy,gc,json,time,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_onset_prior_context as prior
from onset_interval_core import endpoint_allowed,endpoint_grad,PATTERNS,interval_distribution,softmax,interval_decode
BASE=prior.BASE;OUT=BASE/'outputs/early_onset_20261030';RUN=BASE/'work/early_onset_20261030';SEEDS=prior.SEEDS
CONFIGS=[dict(name='context_timing',head='context_timing'),dict(name='weighted_timing',head='weighted_timing'),dict(name='early_half',head='early',blend=.5),dict(name='early_full',head='early',blend=1.)]
def read(p):return prior.read(p)
def write(p,v):return prior.write(p,v)
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def init():
    OUT.mkdir(exist_ok=True);RUN.mkdir(exist_ok=True)
    protocol=dict(training='Development June-September only; final June-November only. Reuse frozen prior-context event heads; retrain timing only.',features='Same996 causal features:940 base +8 previous-day priors +48 previous-day normal-reference anomalies. No query targets or future masks as inputs.',
        timing=dict(context_timing='Six-class endpoint-type-conditioned start/end,996 context features',weighted_timing='Same timing head, positive sequences with any known positive in first3 horizons receive3x row weight',early='Three-horizon conditional queue classifier on known cells of sequences with observed future event; horizon index as input; missing cells excluded; known noncontiguous sequences included. Blend conditional first3 with old timing by .5 or1; later3 unchanged.'),
        budget=dict(seeds=SEEDS,rounds=400,leaves=31,depth=6,min_leaf=60,l2=15,learning_rate=.04,threads=8),
        October_selection='Fixed .5 decoder, one prespecified seed. Eligible if IoU>=control-.01, FP<=1.15control, totalFN<=1.05control. Among eligible select lowest first3 FN then highest IoU; take top2. If fewer than2 eligible fill by IoU diagnostic-only.',
        November='Top2 with3 seeds versus same-date-trained prior-context control; report IoU/earlyFN/FP and paired-day intervals. No November-driven threshold/config change. Both final candidates proceed as research candidates.',
        January='Two thresholds early/late from .25.. .75 step.05 only. Eligible: IoU>=R127, first3FN<=.95R127, FP<=1.10R127,totalFN<=1.05R127,no road loss>.05. MaxIoU among eligible, tie lower earlyFN thenFP; if none eligible choose diagnostic maxIoU only. Apply same search to unchanged prior control.',
        February='Only after all January selections and model hashes frozen. Evaluate both retained candidates and calibration-only control once on the same two separate populations. No changes from February scores; archive every candidate with either population IoU>R127.',
        caution='The request targets short-horizon errors previously seen on February. All new choices/thresholds are selected using October/November/January only; February is reused development evidence, not an independent blind confirmation.',official_S_queue=None)
    write(OUT/'protocol.json',protocol)
    files=[prior.OUT/'January_selection.json']+list((prior.OUT/'final_reference').glob('*.npz'))+list((prior.OUT/'development_reference').glob('*.npz'))
    for phase in ['development','final']:
        for seed in SEEDS:files += [prior.OUT/f'{phase}_capacity_{seed}/event.txt',prior.timing_file(phase,seed)]
    write(OUT/'source_hashes.json',{str(p.relative_to(BASE)):digest(p) for p in files})
def check_sources():
    for p,h in read(OUT/'source_hashes.json').items():assert digest(BASE/p)==h
def shared(phase,seed,kind):
    folder=OUT/'shared';folder.mkdir(exist_ok=True);target=folder/f'{phase}_{seed}_{kind}.txt'
    source=prior.OUT/f'{phase}_capacity_{seed}/event.txt' if kind=='event' else prior.timing_file(phase,seed)
    if not target.exists():shutil.copy2(source,target)
    assert digest(target)==digest(source);return target
def train_head(head,phase,seed):
    folder=OUT/f'{phase}_{head}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    check_sources();x,y,m,w=prior.training('combined',phase);pos=(y.astype(bool)&m).any(1)
    if head=='early':
        ids,hh=np.nonzero(m[:,:3]&pos[:,None]);tx=np.column_stack([x[ids],hh]).astype('float32');ty=y[ids,hh].astype('float32');tw=w[ids];objective='binary';extra={}
    else:
        allowed,representable=endpoint_allowed(y,m);ids=np.flatnonzero(pos&representable);tx=np.column_stack([np.repeat(x[ids],2,axis=0),np.tile([0.,1.],len(ids))]).astype('float32');aa=allowed[ids].reshape(-1,6);tw=np.repeat(w[ids],2)
        if head=='weighted_timing':tw*=np.repeat(np.where((y[ids,:3].astype(bool)&m[ids,:3]).any(1),3.,1.),2)
        tw/=tw.mean();ty=np.zeros(len(tx));objective=lambda z,d:endpoint_grad(z,aa,tw);extra=dict(num_class=6)
    if head=='early':tw=tw/tw.mean()
    params=dict(objective=objective,learning_rate=.04,num_leaves=31,max_depth=6,min_data_in_leaf=60,lambda_l2=15,feature_fraction=.8,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1,feature_pre_filter=False,**extra)
    def cb(env):
        if (env.iteration+1)%100==0:log(phase=phase,head=head,seed=seed,round=env.iteration+1)
    ds=lgb.Dataset(tx,label=ty,weight=tw if head=='early' else None);model=lgb.train(params,ds,num_boost_round=400,callbacks=[cb]);model.save_model(str(folder/'model.txt'));np.save(folder/'feature_gain.npy',model.feature_importance('gain'));write(folder/'complete.json',dict(phase=phase,head=head,seed=seed,source_rows=len(x),supervised_rows=len(tx),dimensions=tx.shape[1],known_cells_only=True));del x,y,m,w,tx,model,ds;gc.collect()
def query(pop):
    if pop in ['development','january']:return prior.query('combined',pop)
    assert (OUT/'frozen_before_February.json').exists()
    data,meta=prior.REF.load(pop);data['parent']=np.load(prior.REF.RUN/pop/'parent.npy');file=RUN/f'{pop}_X.npy'
    if file.exists():return np.load(file,mmap_mode='r'),data,meta
    index={'new_phase_population':'february-new_phase_population.json','old410':'february-old410.json'}[pop];entries={(e['panel'],prior.old.normtime(e['origin'])):e for e in prior.REF.entries(index)};get,_,rf,_=prior.analog.setup();states={};pp=[];cc=[]
    from onset_context_core import PastContext
    for win in meta['windows']:
        panel=win['panel'];s=get(entries[(panel,prior.old.normtime(win['origin']))])
        if panel not in states:states[panel]=PastContext.load(prior.OUT/'final_reference'/f'{panel}.npz')
        p,c,_,_=states[panel].features(s,rf.ramp_history(s),rf.topology(panel)['counts']);np.testing.assert_array_equal(s['y'].T,data['y'][win['start']:win['end']]);np.testing.assert_array_equal(s['m'].T,data['m'][win['start']:win['end']]);pp.append(p);cc.append(c)
    x=np.column_stack([data['X'],np.concatenate(pp),np.concatenate(cc)]).astype('float32');np.save(file,x);return x,data,meta
def components(phase,seed,x,pop):
    file=RUN/f'{phase}_{seed}_{pop}_components.npz'
    if file.exists():
        with np.load(file) as z:return z['e'],z['q']
    event=lgb.Booster(model_file=str(shared(phase,seed,'event')));timing=lgb.Booster(model_file=str(shared(phase,seed,'timing')));q=prior.old.endpoint_predict(timing,x[:,:940])@PATTERNS;e=np.concatenate([event.predict(x[a:a+16384],raw_score=True,num_threads=8) for a in range(0,len(x),16384)]);e=np.exp(-np.logaddexp(0.,-e));np.savez(file,e=e,q=q);return e,q
def predict(config,phase,seed,x,pop):
    folder=OUT/f"{phase}_{config['head']}_{seed}";file=folder/f'{pop}_probability_{config["name"]}.npy'
    if file.exists():return np.load(file)
    e,q=components(phase,seed,x,pop)
    if config['head']!='control':
        model=lgb.Booster(model_file=str(folder/'model.txt'))
        if config['head']=='early':
            short=[]
            for h in range(3):short.append(np.concatenate([model.predict(np.column_stack([x[a:a+16384],np.full(len(x[a:a+16384]),h)]),num_threads=8) for a in range(0,len(x),16384)]))
            q=q.copy();q[:,:3]=(1-config['blend'])*q[:,:3]+config['blend']*np.stack(short,1)
        else:q=prior.old.endpoint_predict(model,x)@PATTERNS
    p=(e[:,None]*q).astype('float32');assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1+1e-6
    folder.mkdir(exist_ok=True);np.save(file,p);return p
def score(p,data,meta,threshold=.5,decoder='interval'):
    r=prior.old.score(p,data,meta,threshold,decoder);b=interval_decode(p,threshold) if decoder=='interval' else p>=threshold;fn=tp=fp=0;rr={}
    for w in meta['windows']:
        if not w['positive']:continue
        a,z=w['start'],w['end'];y=np.asarray(data['y'][a:z,:3],bool);m=np.asarray(data['m'][a:z,:3],bool);q=b[a:z,:3];ft=int((~q&y&m).sum());tt=int((q&y&m).sum());ff=int((q&~y&m).sum());fn+=ft;tp+=tt;fp+=ff;v=rr.setdefault(w['panel'],[0,0]);v[0]+=tt;v[1]+=tt+ft
    r['early']=dict(FN=fn,TP=tp,FP=fp,recall=tp/max(tp+fn,1),macro_recall=float(np.mean([v[0]/max(v[1],1) for v in rr.values()])));return r
def compact(r):return dict(**prior.compact(r),early=r['early'])
def subset(m,month):return prior.subset(m,month)
def screen():
    init();x,d,m=query('development');october=subset(m,'2030-10');control=predict(dict(name='control',head='control'),'development',SEEDS[0],x,'development');base=score(control,d,october);write(OUT/'October_control.json',base);del x;results=[]
    for cfg in CONFIGS:
        train_head(cfg['head'],'development',SEEDS[0]);x,d,m=query('development');p=predict(cfg,'development',SEEDS[0],x,'development');s=score(p,d,subset(m,'2030-10'));eligible=s['onset_IoU']>=base['onset_IoU']-.01 and s['cell_counts']['FP_cells']<=1.15*base['cell_counts']['FP_cells'] and s['cell_counts']['FN_cells']<=1.05*base['cell_counts']['FN_cells'];results.append(dict(config=cfg,eligible=bool(eligible),October=s));write(OUT/'October_search.json',results);log(phase='OCTOBER',config=cfg['name'],eligible=bool(eligible),**compact(s));del x;gc.collect()
    good=sorted([r for r in results if r['eligible']],key=lambda r:(r['October']['early']['FN'],-r['October']['onset_IoU']));fallback=sorted([r for r in results if not r['eligible']],key=lambda r:-r['October']['onset_IoU']);top=(good+fallback)[:2];write(OUT/'selected_candidates.json',top);log(phase='SCREEN_COMPLETE',candidates=[r['config']['name'] for r in top])
def confirm():
    reports=[];x,d,m=query('development');nov=subset(m,'2030-11');basep=np.mean([predict(dict(name='control',head='control'),'development',s,x,'development') for s in SEEDS],0);baseline=score(basep,d,nov);del x
    for item in read(OUT/'selected_candidates.json'):
        cfg=item['config'];ps=[]
        for seed in SEEDS:
            train_head(cfg['head'],'development',seed);x,d,m=query('development');ps.append(predict(cfg,'development',seed,x,'development'));del x;gc.collect()
        p=np.mean(ps,0);s=score(p,d,nov);reports.append(dict(config=cfg,November=s,seed_IoU=[score(z,d,nov)['onset_IoU'] for z in ps],IoU_vs_control=prior.REF.bootstrap(s,baseline)));log(phase='NOVEMBER',config=cfg['name'],**compact(s))
    write(OUT/'November_confirmation.json',dict(control=baseline,candidates=reports))
def final_train():
    for item in read(OUT/'selected_candidates.json'):
        for seed in SEEDS:train_head(item['config']['head'],'final',seed);shared('final',seed,'event');shared('final',seed,'timing')
    check_sources();log(phase='FINAL_COMPLETE')
def calibrate():
    x,d,m=query('january');base=prior.old.ar.score(d['parent'],d['y'],d['m'],m);r127=score(np.asarray(d['parent']),d,m,decoder='cells');assert base['onset_IoU']==r127['onset_IoU'];reports={};configs=[dict(name='control',head='control')]+[r['config'] for r in read(OUT/'selected_candidates.json')]
    for cfg in configs:
        ps=[predict(cfg,'final',s,x,'january') for s in SEEDS];p=np.mean(ps,0);np.save(OUT/f"january_{cfg['name']}_probability.npy",p);grid=[];best=None;best_any=None
        for te in np.round(np.arange(.25,.751,.05),2):
            for tl in np.round(np.arange(.25,.751,.05),2):
                threshold=[float(te)]*3+[float(tl)]*3;s=score(p,d,m,np.array(threshold));c=s['cell_counts'];b=base['cell_counts'];eligible=s['onset_IoU']>=base['onset_IoU'] and s['early']['FN']<=.95*r127['early']['FN'] and c['FP_cells']<=1.10*b['FP_cells'] and c['FN_cells']<=1.05*b['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.05
                record=dict(threshold=threshold,eligible=bool(eligible),**compact(s));grid.append(record);rank=(s['onset_IoU'],-s['early']['FN'],-c['FP_cells'])
                if best_any is None or rank>best_any[0]:best_any=(rank,record,s)
                if eligible and (best is None or rank>best[0]):best=(rank,record,s)
        chosen=(best or best_any);reports[cfg['name']]=dict(config=cfg,selection=chosen[1],feasible_threshold_exists=best is not None,January=chosen[2],IoU_vs_R127=prior.REF.bootstrap(chosen[2],base),grid=grid);log(phase='JANUARY',config=cfg['name'],feasible=best is not None,threshold=chosen[1]['threshold'],**compact(chosen[2]))
    write(OUT/'January_results.json',dict(R127=r127,candidates=reports));check_sources();files=list((OUT/'shared').glob('*.txt'))+list(OUT.glob('final_*/model.txt'))+[OUT/'January_results.json',OUT/'selected_candidates.json',OUT/'protocol.json',OUT/'source_hashes.json'];write(OUT/'frozen_before_February.json',{p.relative_to(OUT).as_posix():digest(p) for p in files});log(phase='FROZEN')
def evaluate():
    frozen=read(OUT/'frozen_before_February.json')
    for p,h in frozen.items():assert digest(OUT/p)==h
    check_sources();cal=read(OUT/'January_results.json');reports={}
    for pop in ['new_phase_population','old410']:
        x,d,m=query(pop);baseline=score(d['parent'],d,m,decoder='cells');out={}
        for name,r in cal['candidates'].items():
            cfg=r['config'];ps=[predict(cfg,'final',s,x,pop) for s in SEEDS];p=np.mean(ps,0);np.save(OUT/f'{pop}_{name}_probability.npy',p);s=score(p,d,m,np.array(r['selection']['threshold']));out[name]=dict(score=s,threshold=r['selection']['threshold'],IoU_vs_R127=prior.REF.bootstrap(s,baseline),exceeds_R127_IoU=s['onset_IoU']>baseline['onset_IoU']);log(phase='FEBRUARY',population=pop,config=name,**compact(s))
        reports[pop]=dict(R127=baseline,candidates=out);write(OUT/'February_results.json',reports);del x;gc.collect()
    for p,h in frozen.items():assert digest(OUT/p)==h
    log(phase='EVALUATION_COMPLETE')
if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['screen','confirm','train-final','calibrate','evaluate']);action=ap.parse_args().action;{'screen':screen,'confirm':confirm,'train-final':final_train,'calibrate':calibrate,'evaluate':evaluate}[action]()
