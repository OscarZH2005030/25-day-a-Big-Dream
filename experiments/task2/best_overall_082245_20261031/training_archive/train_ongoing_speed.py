"""Speed-distribution ongoing experiment. January-only thresholds, frozen best onset."""
import argparse,copy,gc,json,time,hashlib,shutil,sys,subprocess,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import train_ongoing as old
from speed_distribution_core import spatial_features,cdf_zero,odds_mix
BASE=old.BASE;OUT=BASE/'outputs/ongoing_speed_distribution_20261030';RUN=BASE/'work/ongoing_speed_distribution_20261030';SEEDS=old.SEEDS;QUANTILES=[.1,.5,.9]
CONFIGS=[dict(name=f'{head}_{mix}',head=head,mix=mix) for head in ['binary_fields','speed_margin','speed_change'] for mix in ['independent','odds50']]
def read(p):return old.read(p)
def write(p,v):return old.write(p,v)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log(**v):print(json.dumps(dict(seconds=round(time.time()-START,1),**v)),flush=True)
def init():
    OUT.mkdir(exist_ok=True);RUN.mkdir(exist_ok=True)
    write(OUT/'protocol.json',dict(method='Predict future normalized speed margin distribution or speed change, then P(margin<=0); compare matched binary rich-field control. Not queue-flip training.',
      base='Same203 ongoing causal features and exact same node/horizon samples, masks,IPW as previous experiment. Add71 causal whole-road field descriptors and explicit panel indicators; total274 +horizon.',
      fields='Latest reliable speed/load/occupancy within15min, eight released position-basis coefficients perfield and T30-change, coverage,field10/50/90percentiles,panel onehot. No order-index adjacency invented.',
      targets='Reliable future(speed-threshold)/free_speed, unknown cells excluded. Speed-change variant subtracts latest reliable current margin(age<=15), otherwise offset0 with missing-state features explicit.',
      regression='Three quantile heads .10/.50/.90, sorted at prediction; fixed piecewise-linear CDF at0 with fixed exponential tails and width floor.02. No probability calibration or quantile-shape tuning on January/February.',
      budget=dict(rounds=400,leaves=31,depth=6,min_leaf=100,L2=20,learning_rate=.04,maxbin=127,seeds=SEEDS,threads=8),
      October='June-Sep fitting only,end-Sep reference. One fixed seed,6 candidates: binary/absolute margin/change each independent or fixed equal log-odds mixture with R127. At fixed.25 choose top2 IoU with FP/FN<=1.15R127, fill diagnostic highest if needed.',
      November='Top2 configurations with3 seeds, report all; no November reselection. R127 itself fitted Oct-Nov, so anchored development not fully out of sample.',
      final='June-November fitting,unchanged end-Nov reference. Original0.817962444 version and early_full onset weights/thresholds unchanged.',
      January='Only scalar threshold .05.. .75 step.025. MaxIoU subject FP/FN<=1.05R127,no road loss>.03; if none, diagnostic maxIoU. No January model,feature,mix,or quantile-parameter choice.',
      February='After both candidates models+January thresholds hash frozen, evaluate original same2ongoing populations once. No February tuning or candidate replacement. Compare incumbent perpopulation; keep all leaders and report tradeoffs.',
      caveat='Reused development February,not a fresh blind test; reliable observation proxy labels,not official S_queue.',official_S_queue=None))
    shutil.copy2(old.OUT/'frozen_onset_hashes.json',OUT/'frozen_onset_hashes.json');shutil.copy2(old.OUT/'overall_leaders.json',OUT/'incumbent.json')
def check_onset():
    for p,h in read(OUT/'frozen_onset_hashes.json').items():assert digest(BASE/p)==h
def prepare():
    init();get,arrays,network,bases,rf,panels,raw=old.setup();entries={(e['panel'],old.prior.old.normtime(e['origin'])):e for e in raw if e['condition']==1};names=None
    for panel in panels:
        folder=RUN/'fit'/panel;folder.mkdir(exist_ok=True,parents=True)
        if (folder/'complete.json').exists():continue
        source=old.RUN/'fit'/panel;meta=read(source/'complete.json');y=np.load(source/'y.npy',mmap_mode='r');m=np.load(source/'m.npy',mmap_mode='r');xx=[];targets=[];checks=[]
        for w in meta['windows']:
            e=entries[(panel,old.prior.old.normtime(w['origin']))];s=get(e);extra,names=spatial_features(s,bases[panel]['position']);ids=np.array(w['selected_links']);xx.append(extra[ids]);values=arrays(panel,e['origin'][:10]);future=values['speed'][e['origin_slot']+1:e['origin_slot']+7].T;target=((future.astype(float)-s['network']['threshold'][:,None])/s['network']['free_speed'][:,None]).astype('float32');target=np.where(s['m'].T,target,np.nan);a,b=w['start'],w['end'];np.testing.assert_array_equal(s['y'].T[ids],y[a:b]);np.testing.assert_array_equal(s['m'].T[ids],m[a:b]);np.testing.assert_array_equal((target[ids]<=0)[m[a:b]],y[a:b][m[a:b]].astype(bool));targets.append(target[ids])
            if not checks:
                altered=copy.deepcopy(s);altered['y']=1-s['y'];altered['m']=~s['m'];altered['speed_y']=np.ones_like(s['speed_y'])*999;other,nn=spatial_features(altered,bases[panel]['position']);np.testing.assert_array_equal(extra,other);checks.append(dict(panel=panel,future_targets_not_features=True))
        np.save(folder/'extra.npy',np.concatenate(xx));np.save(folder/'margin.npy',np.concatenate(targets));write(folder/'complete.json',dict(rows=meta['rows'],extra_columns=len(names),checks=checks));log(phase='PREPARE_PANEL',panel=panel,rows=meta['rows'],extra=len(names))
    write(OUT/'feature_names.json',dict(base=read(old.OUT/'feature_names.json')['features'],extra=names or spatial_features(get(next(iter(entries.values()))),bases[next(iter(entries.values()))['panel']]['position'])[1],horizon='horizon_minutes'))
    prepare_query('development');log(phase='PREPARED')
def prepare_query(pop):
    if pop not in ['development','january']:assert (OUT/'frozen_before_February.json').exists()
    folder=RUN/pop;folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    source,meta=old.load(pop);get,arrays,network,bases,rf,panels,raw=old.setup();extras=[];targets=[];expected=read(OUT/'feature_names.json')['extra']
    for w in meta['windows']:
        e=w['entry'];s=get(e);x,n=spatial_features(s,bases[e['panel']]['position']);assert n==expected;extras.append(x);values=arrays(e['panel'],e['origin'][:10]);future=values['speed'][e['origin_slot']+1:e['origin_slot']+7].T;target=((future.astype(float)-s['network']['threshold'][:,None])/s['network']['free_speed'][:,None]).astype('float32');targets.append(np.where(s['m'].T,target,np.nan));np.testing.assert_array_equal(s['m'].T,source['m'][w['start']:w['end']]);np.testing.assert_array_equal(s['y'].T,source['y'][w['start']:w['end']])
    np.save(folder/'extra.npy',np.concatenate(extras));np.save(folder/'margin.npy',np.concatenate(targets));write(folder/'complete.json',dict(rows=meta['rows'],windows=len(meta['windows']),future_labels_not_features=True));log(phase='QUERY_PREPARED',population=pop)
def query(pop):
    d,m=old.load(pop);x=np.column_stack([d['X'],np.load(RUN/pop/'extra.npy',mmap_mode='r')]).astype('float32');d=dict(d,X=x);return d,m
def training(phase):
    chunks={k:[] for k in ['X','y','margin','w','offset']};names=read(OUT/'feature_names.json')['base'];speed_i=names.index('speed_latest');known_i=names.index('current_state_known');cut_i=names.index('static_threshold');free_i=names.index('static_free_speed')
    for panel in read(old.RUN/'fit_complete.json')['panels']:
        folder=old.RUN/'fit'/panel;meta=read(folder/'complete.json');base=np.load(folder/'X.npy',mmap_mode='r');extra=np.load(RUN/'fit'/panel/'extra.npy',mmap_mode='r');target=np.load(RUN/'fit'/panel/'margin.npy',mmap_mode='r');y=np.load(folder/'y.npy',mmap_mode='r');m=np.load(folder/'m.npy',mmap_mode='r');h=np.load(folder/'H.npy');w=np.load(folder/'w.npy').copy();use=np.load(folder/'month.npy')<203010 if phase=='development' else np.ones(len(base),bool);wins=[e for e in meta['windows'] if phase!='development' or e['origin'][:7]<'2030-10'];counts={b:sum(e['positive']==b for e in wins) for b in [False,True]}
        for e in wins:w[e['start']:e['end']]*=(.8 if e['positive'] else .2)/counts[e['positive']]
        ids,slot=np.nonzero((h>=0)&use[:,None]);hs=h[ids,slot];x=np.column_stack([base[ids],extra[ids],(hs+1)*5]).astype('float32');offset=np.where(base[ids,known_i]>.5,base[ids,speed_i]-base[ids,cut_i]/base[ids,free_i],0.);weight=w[ids]*m.sum(1)[ids]/(h>=0).sum(1)[ids];yy=target[ids,hs];assert np.isfinite(yy).all();chunks['X'].append(x);chunks['y'].append(y[ids,hs]);chunks['margin'].append(yy);chunks['w'].append(weight);chunks['offset'].append(offset)
    d={k:np.concatenate(v) for k,v in chunks.items()};d['w']/=d['w'].mean();return d
def train(head,phase,seed):
    folder=OUT/f'{phase}_{head}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    d=training(phase);params=dict(learning_rate=.04,num_leaves=31,max_depth=6,min_data_in_leaf=100,lambda_l2=20,max_bin=127,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,num_threads=8,seed=seed,deterministic=True,force_col_wise=True,verbosity=-1);stats=[]
    if head=='cdf_fields':
        cuts=np.random.default_rng(20261102).choice(np.array([-.15,0.,.15],dtype='float32'),len(d['X']));d['X']=np.column_stack([d['X'],cuts]).astype('float32');cdf_label=(d['margin']<=cuts).astype('float32');params['monotone_constraints']=[0]*(d['X'].shape[1]-1)+[1]
    for quantile in ([None] if head in ['binary_fields','cdf_fields'] else QUANTILES):
        label=(cdf_label if head=='cdf_fields' else d['y']) if quantile is None else d['margin']-(d['offset'] if head=='speed_change' else 0);p=dict(params,objective='binary' if quantile is None else 'quantile')
        if quantile is not None:p['alpha']=quantile
        def cb(env):
            if (env.iteration+1)%100==0:log(phase=phase,head=head,seed=seed,quantile=quantile,round=env.iteration+1)
        model=lgb.train(p,lgb.Dataset(d['X'],label=label,weight=d['w']),num_boost_round=400,callbacks=[cb]);tag=('cdf' if head=='cdf_fields' else 'binary') if quantile is None else f'q{int(quantile*100):02d}';model.save_model(str(folder/f'{tag}.txt'));np.save(folder/f'{tag}_gain.npy',model.feature_importance('gain'));del model;gc.collect()
    write(folder/'complete.json',dict(head=head,phase=phase,seed=seed,rows=len(d['X']),dimensions=d['X'].shape[1],rounds=400));del d;gc.collect()
def predict(head,phase,seed,d,pop):
    folder=OUT/f'{phase}_{head}_{seed}';file=folder/f'{pop}_probability.npy'
    if file.exists():return np.load(file)
    tags=['cdf'] if head=='cdf_fields' else ['binary'] if head=='binary_fields' else ['q10','q50','q90'];models=[lgb.Booster(model_file=str(folder/f'{tag}.txt')) for tag in tags];x=d['X'];result=[];qq=[];names=read(OUT/'feature_names.json')['base'];off=np.where(x[:,names.index('current_state_known')]>.5,x[:,names.index('speed_latest')]-x[:,names.index('static_threshold')]/x[:,names.index('static_free_speed')],0.)
    for h in range(6):
        heads=[]
        for model in models:heads.append(np.concatenate([model.predict(np.column_stack([x[a:a+16384],np.full(len(x[a:a+16384]),(h+1)*5)]+([np.zeros(len(x[a:a+16384]))] if head=='cdf_fields' else [])),num_threads=8) for a in range(0,len(x),16384)]))
        if head in ['binary_fields','cdf_fields']:result.append(heads[0])
        else:
            q=np.stack(heads,-1)+(off[:,None] if head=='speed_change' else 0);result.append(cdf_zero(q));qq.append(q.astype('float32'))
    p=np.stack(result,1).astype('float32');assert np.isfinite(p).all();np.save(file,p)
    if qq:np.save(folder/f'{pop}_quantiles.npy',np.stack(qq,1))
    return p
def combine(cfg,p,d):return p if cfg['mix']=='independent' else odds_mix(d['parent'],p)
def probability(cfg,phase,d,pop):return combine(cfg,np.mean([predict(cfg['head'],phase,s,d,pop) for s in SEEDS],0),d)
def score(p,d,m,t=.25):return old.score(p,d,m,t)
def screen():
    d,m=query('development');october=old.subset(m,'2030-10');base=score(d['parent'],d,october);results=[]
    for head in ['binary_fields','speed_margin','speed_change']:
        train(head,'development',SEEDS[0]);p=predict(head,'development',SEEDS[0],d,'development')
        for cfg in [c for c in CONFIGS if c['head']==head]:
            s=score(combine(cfg,p,d),d,october);b=base['cell_counts'];c=s['cell_counts'];ok=c['FP_cells']<=1.15*b['FP_cells'] and c['FN_cells']<=1.15*b['FN_cells'];results.append(dict(config=cfg,eligible=bool(ok),October=s));write(OUT/'October_search.json',results);log(phase='OCTOBER',config=cfg['name'],eligible=bool(ok),**old.compact(s))
    top=sorted(results,key=lambda r:(r['eligible'],r['October']['onset_IoU']),reverse=True)[:2];write(OUT/'selected_candidates.json',top);log(phase='SCREEN_COMPLETE',candidates=[r['config']['name'] for r in top])
def confirm():
    d,m=query('development');nov=old.subset(m,'2030-11');base=score(d['parent'],d,nov);reports=[]
    for item in read(OUT/'selected_candidates.json'):
        cfg=item['config']
        for seed in SEEDS:train(cfg['head'],'development',seed)
        p=probability(cfg,'development',d,'development');s=score(p,d,nov);reports.append(dict(config=cfg,November=s,IoU_vs_R127=old.prior.REF.bootstrap(s,base)));log(phase='NOVEMBER',config=cfg['name'],**old.compact(s))
    write(OUT/'November_confirmation.json',dict(R127=base,candidates=reports))
def final_train():
    jobs=[(head,seed) for head in sorted({item['config']['head'] for item in read(OUT/'selected_candidates.json')}) for seed in SEEDS]
    write(OUT/'execution_notes.json',dict(final_training_max_concurrent_processes=2,threads_per_process=8,model_parameters_and_seeds_unchanged=True,distinct_output_folder_per_head_and_seed=True))
    def fit(job):
        head,seed=job;subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'fit','--head',head,'--phase','final','--seed',str(seed)],check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(fit,jobs))
    check_onset();prepare_query('january');log(phase='FINAL_COMPLETE')
def calibrate():
    check_onset();d,m=query('january');base=score(d['parent'],d,m);reports={}
    for item in read(OUT/'selected_candidates.json'):
        cfg=item['config'];p=probability(cfg,'final',d,'january');np.save(OUT/f"january_{cfg['name']}_probability.npy",p);grid=[];best=None;unconstrained=None
        for threshold in np.round(np.arange(.05,.7501,.025),3):
            s=score(p,d,m,float(threshold));c=s['cell_counts'];b=base['cell_counts'];ok=c['FP_cells']<=1.05*b['FP_cells'] and c['FN_cells']<=1.05*b['FN_cells'] and min(s['per_road'][r]-base['per_road'][r] for r in base['per_road'])>=-.03;record=dict(threshold=float(threshold),eligible=bool(ok),**old.compact(s));grid.append(record)
            if unconstrained is None or s['onset_IoU']>unconstrained[1]['onset_IoU']:unconstrained=(record,s)
            if ok and (best is None or s['onset_IoU']>best[1]['onset_IoU']):best=(record,s)
        chosen=best or unconstrained;reports[cfg['name']]=dict(config=cfg,threshold=chosen[0]['threshold'],feasible=best is not None,January=chosen[1],grid=grid,IoU_vs_R127=old.prior.REF.bootstrap(chosen[1],base));log(phase='JANUARY',config=cfg['name'],threshold=chosen[0]['threshold'],feasible=best is not None,**old.compact(chosen[1]))
    write(OUT/'January_results.json',dict(R127=base,candidates=reports));files=[OUT/'January_results.json',OUT/'selected_candidates.json',OUT/'protocol.json',OUT/'feature_names.json',OUT/'incumbent.json',OUT/'frozen_onset_hashes.json']+list(OUT.glob('*amendment*.json'))+list(OUT.glob('final_*/*.txt'));write(OUT/'frozen_before_February.json',{p.relative_to(OUT).as_posix():digest(p) for p in files});log(phase='FROZEN')
def evaluate():
    for p,h in read(OUT/'frozen_before_February.json').items():assert digest(OUT/p)==h
    check_onset();cal=read(OUT/'January_results.json');onset=read(BASE/'outputs/early_onset_20261030/February_results.json');incumbent=read(OUT/'incumbent.json')['best_observed_by_population'];reports={}
    for pop in ['new_phase_population','old410']:
        prepare_query(pop);d,m=query(pop);base=score(d['parent'],d,m);candidates={};fixed=onset[pop]['candidates']['early_full']['score']['onset_IoU']
        for name,r in cal['candidates'].items():
            p=probability(r['config'],'final',d,pop);np.save(OUT/f'{pop}_{name}_probability.npy',p);s=score(p,d,m,r['threshold']);candidates[name]=dict(score=s,threshold=r['threshold'],IoU_vs_R127=old.prior.REF.bootstrap(s,base),fixed_onset_IoU=fixed,overall_IoU=(s['onset_IoU']+fixed)/2,exceeds_current_best=s['onset_IoU']>incumbent[pop]['ongoing_IoU']);log(phase='FEBRUARY',population=pop,config=name,overall=candidates[name]['overall_IoU'],exceeds_current_best=candidates[name]['exceeds_current_best'],**old.compact(s))
        reports[pop]=dict(R127=base,incumbent=incumbent[pop],candidates=candidates);write(OUT/'February_results.json',reports);del d;gc.collect()
    check_onset();log(phase='EVALUATION_COMPLETE')
if __name__=='__main__':
    START=time.time();ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','screen','confirm','train-final','calibrate','evaluate','fit']);ap.add_argument('--head');ap.add_argument('--phase');ap.add_argument('--seed',type=int);args=ap.parse_args()
    if args.action=='fit':train(args.head,args.phase,args.seed)
    else:{'prepare':prepare,'screen':screen,'confirm':confirm,'train-final':final_train,'calibrate':calibrate,'evaluate':evaluate}[args.action]()
